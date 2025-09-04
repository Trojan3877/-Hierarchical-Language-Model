# src/evaluate.py

"""
Evaluation Script for Hierarchical Language Model (HLM)
-------------------------------------------------------
Provides functions to evaluate the trained model on validation
or test datasets. Includes metrics like accuracy, precision,
recall, and F1 score.
"""

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from train import HLM


def evaluate_model(model, data, labels):
    """
    Evaluates the model on validation/test data.

    Args:
        model (nn.Module): Trained HLM model
        data (list): Nested list of documents [batch, paragraphs, sentences]
        labels (torch.Tensor): Ground truth labels

    Returns:
        dict: Dictionary of evaluation metrics
    """
    model.eval()
    with torch.no_grad():
        outputs = model(data)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        true_labels = labels.cpu().numpy()

    metrics = {
        "accuracy": accuracy_score(true_labels, predictions),
        "precision": precision_score(true_labels, predictions, average="weighted"),
        "recall": recall_score(true_labels, predictions, average="weighted"),
        "f1": f1_score(true_labels, predictions, average="weighted")
    }

    return metrics


# -----------------
# Demo Run
# -----------------
if __name__ == "__main__":
    # Dummy dataset: 1 doc, 2 paragraphs, 2 sentences each
    documents = [
        [
            ["The food was delicious.", "Service was excellent."],
            ["I would come back again.", "Highly recommended."]
        ]
    ]
    labels = torch.tensor([1])  # Example: positive sentiment

    model = HLM(num_classes=2)
    metrics = evaluate_model(model, documents, labels)

    print("Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
