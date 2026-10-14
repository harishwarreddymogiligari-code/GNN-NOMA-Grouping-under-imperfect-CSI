import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import Dict, Any, Tuple


def align_cluster_labels(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Aligns predicted group indices with true class labels using the Hungarian
    algorithm (linear sum assignment) to resolve label permutation ambiguity.
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        return y_pred

    # Filter out unlabeled samples (-1)
    mask = (y_true != -1) & (y_pred != -1)
    if not mask.any():
        return y_pred

    y_true_valid = y_true[mask]
    y_pred_valid = y_pred[mask]

    num_classes = max(int(np.max(y_true_valid)), int(np.max(y_pred_valid))) + 1
    cost_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    for i in range(len(y_true_valid)):
        cost_matrix[y_pred_valid[i], y_true_valid[i]] += 1

    # Maximize total overlap (minimize negative cost matrix)
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)

    mapping = {row: col for row, col in zip(row_ind, col_ind)}
    
    aligned_preds = np.array([mapping.get(val, val) if val != -1 else -1 for val in y_pred])
    return aligned_preds


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    align_labels: bool = False
) -> Dict[str, Any]:
    """
    Computes comprehensive classification metrics: Accuracy, Precision, Recall,
    F1-score, and Confusion Matrix.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mask = (y_true != -1)
    if not mask.any():
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "confusion_matrix": np.array([])
        }

    yt = y_true[mask]
    yp = y_pred[mask]

    if align_labels:
        yp = align_cluster_labels(yt, yp)

    acc = float(accuracy_score(yt, yp))
    precision, recall, f1, _ = precision_recall_fscore_support(yt, yp, average='macro', zero_division=0)
    cm = confusion_matrix(yt, yp)

    return {
        "accuracy": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm
    }
