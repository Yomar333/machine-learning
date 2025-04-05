################################################################################
# Image-Text Matching Classifier for Multi-Choice Visual Question Answering
#
# This program implements classifiers for multi-choice VQA, comparing different
# vision and text encoders, loss functions, optimizers, and hyperparameters.
# It uses Optuna for hyperparameter optimization and integrates a large language model.
# The ResNet50 vision encoder has been used for pretrained models, and ResNet50-Scratch
# has been replaced with EfficientNet-Scratch.
#
# Version 2.2, created by Grok 3 (xAI)
# Acknowledgments:
# - PyTorch and torchvision (Paszke et al., 2019)
# - HuggingFace Transformers (Wolf et al., 2020)
# - Optuna (Akiba et al., 2019)
################################################################################

import os
import time
import pickle
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer
import optuna
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
from torchvision.models import vit_b_32

# Set random seeds for reproducibility
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# Custom Dataset for Multi-Choice VQA
class VQA_Dataset(Dataset):
    def __init__(self, images_path, data_file, data_split, model_type="ResNet50", train_ratio=0.7, val_ratio=0.15):
        self.images_path = images_path
        self.data_file = data_file
        self.data_split = data_split.lower()
        self.model_type = model_type.upper()
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        self.image_data = []
        self.question_data = []
        self.candidates_data = []  # List of candidate answers
        self.label_data = []  # Index of the correct candidate

        self.transform = self.get_model_specific_transform()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.load_data()

    def get_model_specific_transform(self):
        base_transforms = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        # Add data augmentation for training
        if self.data_split == "train":
            base_transforms.insert(0, transforms.RandomHorizontalFlip())
            base_transforms.insert(0, transforms.RandomRotation(10))

        if self.model_type == "VIT":
            return transforms.Compose([transforms.Resize((224, 224))] + base_transforms)
        elif self.model_type == "EFFICIENTNET":
            return transforms.Compose([transforms.Resize((224, 224))] + base_transforms)  # EfficientNet-B0 uses 224x224
        elif self.model_type == "INCEPTIONV3":
            return transforms.Compose([transforms.Resize((299, 299))] + base_transforms)
        elif self.model_type in ["RESNET50", "RESNET50_BERT"]:
            return transforms.Compose([transforms.Resize((224, 224))] + base_transforms)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def load_data(self):
        print(f"LOADING data from {self.data_file} for {self.model_type}")
        print("=========================================")
        with open(self.data_file) as f:
            lines = f.readlines()
            if self.data_split in ["train", "val"]:
                random.shuffle(lines)
                total_samples = len(lines)
                train_samples = int(total_samples * self.train_ratio)
                val_samples = int(total_samples * self.val_ratio)
                if self.data_split == "train":
                    lines = lines[:train_samples]
                elif self.data_split == "val":
                    lines = lines[train_samples:train_samples + val_samples]

            for line in lines:
                line = line.rstrip("\n")
                parts = line.split("\t")
                img_name = parts[0].strip()
                img_path = os.path.join(self.images_path, img_name)
                question = parts[1].strip()
                candidates = [parts[i].strip() for i in range(2, len(parts) - 1)]  # Multiple candidates
                correct_idx = int(parts[-1])  # Index of correct candidate

                self.image_data.append(img_path)
                self.question_data.append(question)
                self.candidates_data.append(candidates)
                self.label_data.append(correct_idx)

        print(f"|image_data|={len(self.image_data)}")
        print("done loading data...")

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        img_path = self.image_data[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        question = self.question_data[idx]
        candidates = self.candidates_data[idx]
        label = torch.tensor(self.label_data[idx], dtype=torch.long)

        # Tokenize question and candidates
        question_inputs = self.tokenizer(question, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
        candidate_inputs = [self.tokenizer(cand, return_tensors="pt", padding="max_length", truncation=True, max_length=128) for cand in candidates]

        return img, question_inputs, candidate_inputs, label

# Focal Loss for handling class imbalance
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction="none")(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss

# Vision Encoder (ResNet50, EfficientNet, or ViT)
class VisionEncoder(nn.Module):
    def __init__(self, architecture="ResNet50", pretrained=True, train_from_scratch=False):
        super(VisionEncoder, self).__init__()
        self.architecture = architecture
        if architecture == "ResNet50":
            self.model = models.resnet50(pretrained=pretrained and not train_from_scratch)
            if pretrained and not train_from_scratch:
                for param in self.model.parameters():
                    param.requires_grad = False
                for param in list(self.model.children())[-2:]:
                    for p in param.parameters():
                        p.requires_grad = True
            self.model.fc = nn.Linear(self.model.fc.in_features, 512)
        elif architecture == "EfficientNet":
            self.model = models.efficientnet_b0(pretrained=pretrained and not train_from_scratch)
            if pretrained and not train_from_scratch:
                for param in self.model.parameters():
                    param.requires_grad = False
                for param in list(self.model.classifier.parameters())[-2:]:
                    param.requires_grad = True
            self.model.classifier = nn.Linear(self.model.classifier[1].in_features, 512)
        elif architecture == "ViT":
            self.model = vit_b_32(weights="IMAGENET1K_V1" if pretrained and not train_from_scratch else None)
            if pretrained and not train_from_scratch:
                for param in self.model.parameters():
                    param.requires_grad = False
                for param in list(self.model.heads.parameters())[-2:]:
                    param.requires_grad = True
            self.num_features = self.model.heads[0].in_features
            self.model.heads = nn.Identity()
            self.fc = nn.Linear(self.num_features, 512)

    def forward(self, x):
        features = self.model(x)
        if self.architecture == "ViT":
            features = self.fc(features)
        return features

# Text Encoder (Trainable BERT)
class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(self.bert.config.hidden_size, 512)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids.squeeze(1), attention_mask=attention_mask.squeeze(1))
        cls_token = outputs.last_hidden_state[:, 0, :]  # CLS token
        return self.fc(cls_token)

# VQA Model
class VQA_Model(nn.Module):
    def __init__(self, num_candidates=4, vision_arch="ResNet50", pretrained=True, train_from_scratch=False):
        super(VQA_Model, self).__init__()
        self.vision_encoder = VisionEncoder(architecture=vision_arch, pretrained=pretrained, train_from_scratch=train_from_scratch)
        self.text_encoder = TextEncoder()
        self.fc = nn.Linear(512 + 512, 512)
        self.classifier = nn.Linear(512, num_candidates)

    def forward(self, img, question_inputs, candidate_inputs):
        # Vision features
        img_features = self.vision_encoder(img)

        # Question features
        question_features = self.text_encoder(question_inputs["input_ids"], question_inputs["attention_mask"])

        # Candidate scores
        combined = torch.cat((img_features, question_features), dim=1)
        combined = self.fc(combined)
        combined = torch.relu(combined)

        # Score each candidate
        scores = self.classifier(combined)
        return scores

# Training Function with Early Stopping
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    print(f'TRAINING {model.__class__.__name__} model')
    model.train()
    best_val_loss = float("inf")
    patience = 3
    patience_counter = 0

    train_times = []
    for epoch in range(num_epochs):
        start_time = time.time()
        running_loss = 0.0
        for images, question_inputs, candidate_inputs, labels in train_loader:
            images = images.to(device)
            question_inputs = {k: v.to(device) for k, v in question_inputs.items()}
            labels = labels.to(device)

            outputs = model(images, question_inputs, candidate_inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        val_loss = evaluate_model(model, val_loader, criterion, device, mode="val")
        elapsed_time = time.time() - start_time
        train_times.append(elapsed_time)

        print(f'Epoch [{epoch + 1}/{num_epochs}] Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {elapsed_time:.2f} seconds')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    return sum(train_times) / len(train_times)

# Evaluation Function
def evaluate_model(model, loader, criterion, device, mode="test"):
    model.eval()
    total_loss = 0
    all_labels = []
    all_predictions = []
    mrr_scores = []
    start_time = time.time()

    with torch.no_grad():
        for images, question_inputs, candidate_inputs, labels in loader:
            images = images.to(device)
            question_inputs = {k: v.to(device) for k, v in question_inputs.items()}
            labels = labels.to(device)

            outputs = model(images, question_inputs, candidate_inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predicted_class = outputs.argmax(dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted_class.cpu().numpy())

            # MRR calculation
            softmax_scores = torch.softmax(outputs, dim=1)
            for i in range(len(labels)):
                sorted_indices = torch.argsort(softmax_scores[i], descending=True)
                rank = (sorted_indices == labels[i]).nonzero(as_tuple=True)[0].item() + 1
                mrr_scores.append(1.0 / rank)

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    if mode == "val":
        return total_loss / len(loader)

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average="macro")
    f1 = f1_score(all_labels, all_predictions, average="macro")
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    mean_mrr = np.mean(mrr_scores)
    eval_time = time.time() - start_time

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Mean MRR: {mean_mrr:.4f}')
    print(f'Confusion Matrix:\n{conf_matrix}')
    print(f'Total Test Loss: {total_loss:.4f}')
    print(f'Evaluation Time: {eval_time:.2f} seconds')

    return {
        "accuracy": accuracy,
        "precision": precision,
        "f1_score": f1,
        "mrr": mean_mrr,
        "confusion_matrix": conf_matrix,
        "eval_time": eval_time,
        "test_loss": total_loss
    }

# Hyperparameter Optimization with Optuna
def objective(trial, train_loader, val_loader, device, vision_arch, pretrained, train_from_scratch):
    # Define hyperparameters to tune
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    num_epochs = trial.suggest_int("num_epochs", 5, 20)
    use_focal_loss = trial.suggest_categorical("use_focal_loss", [True, False])
    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "SGD"])

    model = VQA_Model(num_candidates=4, vision_arch=vision_arch, pretrained=pretrained, train_from_scratch=train_from_scratch).to(device)
    criterion = FocalLoss(gamma=2.0) if use_focal_loss else nn.CrossEntropyLoss()
    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)
    val_loss = evaluate_model(model, val_loader, criterion, device, mode="val")
    return val_loss

# Main Execution
if __name__ == "__main__":
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Paths and files
    IMAGES_PATH = "visual7w-images"
    data_file = "./visual7w-text/v7w.Images.itm.txt"  # Adjusted to a single file for splitting
    for path in [IMAGES_PATH, data_file]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")

    # Datasets
    train_dataset = VQA_Dataset(IMAGES_PATH, data_file, "train", model_type="ResNet50", train_ratio=0.7, val_ratio=0.15)
    val_dataset = VQA_Dataset(IMAGES_PATH, data_file, "val", model_type="ResNet50", train_ratio=0.7, val_ratio=0.15)
    test_dataset = VQA_Dataset(IMAGES_PATH, data_file, "test", model_type="ResNet50")

    # Classifiers to compare
    classifiers = [
        ("ResNet50-Pretrained", {"vision_arch": "ResNet50", "pretrained": True, "train_from_scratch": False}),
        ("EfficientNet-Scratch", {"vision_arch": "EfficientNet", "pretrained": False, "train_from_scratch": True}),
        ("ViT-Pretrained", {"vision_arch": "ViT", "pretrained": True, "train_from_scratch": False}),
    ]

    results = {}
    for name, config in classifiers:
        print(f"\n=== Optimizing {name} ===")
        # Update model_type for dataset based on the classifier
        model_type = "ResNet50" if "ResNet50" in name else "EfficientNet" if "EfficientNet" in name else "ViT"
        train_dataset.model_type = model_type
        val_dataset.model_type = model_type
        test_dataset.model_type = model_type
        train_dataset.transform = train_dataset.get_model_specific_transform()
        val_dataset.transform = val_dataset.get_model_specific_transform()
        test_dataset.transform = test_dataset.get_model_specific_transform()

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

        # Hyperparameter optimization
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, train_loader, val_loader, device, **config), n_trials=10)

        # Train with best hyperparameters
        best_params = study.best_params
        model = VQA_Model(num_candidates=4, **config).to(device)
        criterion = FocalLoss(gamma=2.0) if best_params["use_focal_loss"] else nn.CrossEntropyLoss()
        if best_params["optimizer"] == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=best_params["lr"], weight_decay=1e-4)
        else:
            optimizer = optim.SGD(model.parameters(), lr=best_params["lr"], momentum=0.9)

        train_time = train_model(model, train_loader, val_loader, criterion, optimizer, best_params["num_epochs"], device)
        eval_metrics = evaluate_model(model, test_loader, criterion, device)

        results[name] = {
            "train_time": train_time,
            "best_params": best_params,
            **eval_metrics
        }

    # Print comparison
    print("\nComparison of Classifiers:")
    print("==========================")
    for name, metrics in results.items():
        print(f"\n{name} Results:")
        print(f"Best Hyperparameters: {metrics['best_params']}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"Mean MRR: {metrics['mrr']:.4f}")
        print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
        print(f"Training Time: {metrics['train_time']:.2f} seconds")
        print(f"Evaluation Time: {metrics['eval_time']:.2f} seconds")
        print(f"Test Loss: {metrics['test_loss']:.4f}")

    # Critique and Recommendation
    best_model = max(results.items(), key=lambda x: x[1]["accuracy"])
    print(f"\nBest Model: {best_model[0]} with Accuracy: {best_model[1]['accuracy']:.4f}")
    print("\nCritique and Justification:")
    print("- ResNet50-Pretrained: Leverages transfer learning with ResNet50, expected to perform well due to pre-trained weights on ImageNet.")
    print("- EfficientNet-Scratch: Trained from scratch using EfficientNet-B0, may underperform due to limited data but benefits from efficient scaling of depth, width, and resolution.")
    print("- ViT-Pretrained: Vision Transformer provides strong performance on vision tasks, especially with pre-training.")
    print("Recommendation: The best model is chosen based on accuracy, but ViT-Pretrained may generalize better due to its attention mechanism.")

