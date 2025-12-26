import flwr as fl
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Storage for centralized metrics
central_metrics = {
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1_score": [],
    "loss": []
}

# For simulated confusion matrix (optional per round)
last_labels = []
last_preds = []

def weighted_average(metrics, store=True):
    total_examples = sum(num_examples for num_examples, _ in metrics)

    avg = {
        "accuracy": sum(num_examples * m["accuracy"] for num_examples, m in metrics) / total_examples,
        "precision": sum(num_examples * m["precision"] for num_examples, m in metrics) / total_examples,
        "recall": sum(num_examples * m["recall"] for num_examples, m in metrics) / total_examples,
        "f1_score": sum(num_examples * m["f1_score"] for num_examples, m in metrics) / total_examples,
    }

    if "loss" in metrics[0][1]:
        avg["loss"] = sum(num_examples * m["loss"] for num_examples, m in metrics) / total_examples
        if store:
            central_metrics["loss"].append(avg["loss"])

    if store:
        for key in ["accuracy", "precision", "recall", "f1_score"]:
            central_metrics[key].append(avg[key])

    return avg

def plot_all_metrics():
    os.makedirs("outputs", exist_ok=True)
    rounds = list(range(1, len(central_metrics["accuracy"]) + 1))

    # Combined metrics plot
    plt.figure(figsize=(10, 6))
    for metric in ["accuracy", "precision", "recall", "f1_score"]:
        plt.plot(rounds, central_metrics[metric], marker='o', label=metric.capitalize())
    plt.title("Federated Evaluation Metrics Over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/central_metrics_plot.png")

def plot_loss_curve():
    if central_metrics["loss"]:
        rounds = list(range(1, len(central_metrics["loss"]) + 1))
        plt.figure(figsize=(10, 4))
        plt.plot(rounds, central_metrics["loss"], marker='o', color='blue')
        plt.title("Average Loss per Round")
        plt.xlabel("Round")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.tight_layout()
        os.makedirs("outputs", exist_ok=True)
        plt.savefig("outputs/loss_curve_separate_server.png")
        print("[SERVER] 📉 Loss curve saved.")

def plot_conf_matrix():
    if last_labels and last_preds:
        cm = confusion_matrix(last_labels, last_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Pneumonia"])
        disp.plot(cmap="Blues")
        plt.title("Final Round Confusion Matrix")
        plt.tight_layout()
        os.makedirs("outputs", exist_ok=True)
        plt.savefig("outputs/confusion_matrix_server.png")
        print("[SERVER] 📉 Confusion matrix saved.")

def main():
    def fit_metrics_agg(metrics):
        return weighted_average(metrics, store=False)

    def eval_metrics_agg(metrics):
        global last_labels, last_preds
        if "labels" in metrics[0][1] and "predictions" in metrics[0][1]:
            last_labels = metrics[0][1]["labels"]
            last_preds = metrics[0][1]["predictions"]
        return weighted_average(metrics, store=True)

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        fit_metrics_aggregation_fn=fit_metrics_agg,
        evaluate_metrics_aggregation_fn=eval_metrics_agg,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=20),
        strategy=strategy,
    )

    plot_all_metrics()
    plot_loss_curve()
    plot_conf_matrix()

if __name__ == "__main__":
    main()
