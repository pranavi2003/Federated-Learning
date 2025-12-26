import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
import sys
from model import get_resnet
from dataset import PneumoniaDataset
from opacus import PrivacyEngine
import matplotlib.pyplot as plt
import seaborn as sns
import atexit
from torchvision import transforms

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# Load model
model = get_resnet().to(DEVICE)

# Dataset loading
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

data_path = sys.argv[1]
dataset = PneumoniaDataset(data_path)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Compute class weights
labels = [label for _, label in dataset]
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
weights = torch.tensor(class_weights, dtype=torch.float)

# Loss and optimizer
loss_fn = nn.CrossEntropyLoss(weight=weights.to(DEVICE))
optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

# Privacy engine
privacy_engine = PrivacyEngine()
model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
)

param_norms = []
training_losses = []
eval_metrics = {
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1": []
}
client_id = os.path.basename(data_path.rstrip("/"))

class FLClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        total_norm = torch.norm(torch.stack([torch.norm(p.detach()) for p in model.parameters()]))
        param_norms.append(total_norm.item())
        return [val.cpu().numpy() for val in model.state_dict().values()]

    def set_parameters(self, parameters):
        keys = list(model.state_dict().keys())
        model.load_state_dict(dict(zip(keys, [torch.tensor(p) for p in parameters])))

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        model.train()
        all_preds, all_labels = [], []

        for epoch in range(3):  # <-- 3 local epochs
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(x)
                loss = loss_fn(outputs, y)
                training_losses.append(loss.item())
                loss.backward()
                optimizer.step()
            scheduler.step()

        # Evaluate on train set for reporting
        correct, total = 0, 0
        model.eval()
        with torch.no_grad():
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                outputs = model(x)
                _, preds = torch.max(outputs, 1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        acc = correct / total
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        epsilon = privacy_engine.get_epsilon(delta=1e-5)

        print(f"[{client_id}] 🔐 Epsilon after training: {epsilon:.2f}")

        return self.get_parameters(config), len(train_loader.dataset), {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "loss": np.mean(training_losses[-len(train_loader):])
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        model.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []
        total_loss = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                outputs = model(x)
                loss = loss_fn(outputs, y)
                total_loss += loss.item() * x.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        avg_loss = total_loss / total
        acc = correct / total
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        eval_metrics["accuracy"].append(acc)
        eval_metrics["precision"].append(precision)
        eval_metrics["recall"].append(recall)
        eval_metrics["f1"].append(f1)

        # Confusion Matrix
        os.makedirs("outputs", exist_ok=True)
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Pneumonia"], yticklabels=["Normal", "Pneumonia"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix - {client_id}")
        plt.tight_layout()
        plt.savefig(f"outputs/confusion_matrix_{client_id}.png")

        return avg_loss, total, {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "loss": avg_loss
        }

def plot_metrics():
    os.makedirs("outputs", exist_ok=True)

    # Weight Norms
    plt.figure()
    plt.plot(param_norms, marker='o')
    plt.title(f"Weight Norms - {client_id}")
    plt.xlabel("Round")
    plt.ylabel("L2 Norm")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"outputs/weight_norms_{client_id}.png")

    # Training Loss
    plt.figure()
    plt.plot(training_losses, color='blue')
    plt.title("Training Loss per Batch")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"outputs/training_loss_{client_id}.png")

    # Combined Metrics
    rounds = list(range(1, len(eval_metrics["accuracy"]) + 1))
    plt.figure(figsize=(10, 6))
    for metric in ["accuracy", "precision", "recall", "f1"]:
        plt.plot(rounds, eval_metrics[metric], marker='o', label=metric.capitalize())
    plt.title(f"Evaluation Metrics per Round - {client_id}")
    plt.xlabel("Round")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"outputs/metrics_combined_{client_id}.png")

atexit.register(plot_metrics)

# Start the client
fl.client.start_client(server_address="federated-xray-server:8080", client=FLClient().to_client())
