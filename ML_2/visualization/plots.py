import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import config


def plot_training_history(history: dict, save_dir: str = None):
    """
    Renders 4-panel training curves:
    1. Total Loss vs Epoch
    2. Component Losses (Supervised, Consistency, Pseudo-label) vs Epoch
    3. Validation Accuracy & F1 vs Epoch
    4. Pseudo-label Coverage vs Epoch
    """
    if save_dir is None:
        save_dir = os.path.join(config.RESULTS_DIR, "plots")
    os.makedirs(save_dir, exist_ok=True)

    epochs = history["epoch"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Total & Validation Loss
    axes[0, 0].plot(epochs, history["total_loss"], 'b-', label="Train Loss", linewidth=2)
    axes[0, 0].plot(epochs, history["val_loss"], 'r--', label="Val Loss", linewidth=2)
    axes[0, 0].set_title("Total Training & Validation Loss", fontweight='bold')
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Component Losses
    axes[0, 1].plot(epochs, history["supervised_loss"], 'g-', label="Supervised Loss")
    axes[0, 1].plot(epochs, history["consistency_loss"], 'm-', label="Consistency Loss (λ)")
    axes[0, 1].plot(epochs, history["pseudo_loss"], 'c-', label="Pseudo-Label Loss (μ)")
    axes[0, 1].set_title("Semi-Supervised Loss Decomposition", fontweight='bold')
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss Magnitude")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Validation Accuracy, Precision, Recall & F1
    axes[1, 0].plot(epochs, history["val_accuracy"], 'b-', label="Val Accuracy", linewidth=2)
    if "val_precision" in history and len(history["val_precision"]) > 0:
        axes[1, 0].plot(epochs, history["val_precision"], 'r-.', label="Val Precision", linewidth=1.8)
    if "val_recall" in history and len(history["val_recall"]) > 0:
        axes[1, 0].plot(epochs, history["val_recall"], 'm:', label="Val Recall", linewidth=1.8)
    axes[1, 0].plot(epochs, history["val_f1"], 'g--', label="Val Macro F1", linewidth=2)
    axes[1, 0].set_title("Validation Metrics vs Epoch (Acc, Prec, Rec, F1)", fontweight='bold')
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Pseudo-label Coverage
    axes[1, 1].plot(epochs, history["pseudo_coverage"], 'orange', linewidth=2)
    axes[1, 1].set_title("Pseudo-Label Coverage Ratio on Unlabeled Data", fontweight='bold')
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Coverage Ratio")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[VISUALIZATION] Saved training history curves to: {save_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list = None,
    save_path: str = None,
    metrics: dict = None
):
    """
    Renders an annotated Confusion Matrix heatmap for NOMA user grouping.
    Displays counts, percentages, and global evaluation metrics (Acc, Prec, Rec, F1).
    """
    if save_path is None:
        save_path = os.path.join(config.RESULTS_DIR, "plots", "confusion_matrix.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if cm is None or len(cm) == 0:
        return

    num_classes = cm.shape[0]
    if class_names is None:
        class_names = [f"Group {i+1}" for i in range(num_classes)]

    # Normalized percentages along true rows
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)

    # Annotations formatted with both count and percentage
    annot = np.empty_like(cm, dtype=object)
    for i in range(num_classes):
        for j in range(num_classes):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]:.1%})"

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm,
        annot=annot,
        fmt='',
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
        linewidths=1.0,
        linecolor='white'
    )

    title_text = "NOMA User Grouping Confusion Matrix"
    if metrics is not None:
        acc = metrics.get('accuracy', 0.0)
        prec = metrics.get('precision', 0.0)
        rec = metrics.get('recall', 0.0)
        f1 = metrics.get('f1_score', 0.0)
        title_text += f"\nAcc: {acc:.2%} | Prec: {prec:.2%} | Rec: {rec:.2%} | F1: {f1:.2%}"

    ax.set_title(title_text, fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel("Predicted NOMA Group", fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel("True Ground Truth Group", fontsize=11, fontweight='bold', labelpad=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[VISUALIZATION] Saved confusion matrix heatmap to: {save_path}")


def plot_metrics_summary_bar(metrics: dict, save_path: str = None):
    """
    Renders a bar chart comparing global classification metrics:
    Accuracy, Precision (Macro), Recall (Macro), and F1-Score (Macro).
    """
    if save_path is None:
        save_path = os.path.join(config.RESULTS_DIR, "plots", "metrics_summary.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    metric_values = [
        metrics.get("accuracy", 0.0),
        metrics.get("precision", 0.0),
        metrics.get("recall", 0.0),
        metrics.get("f1_score", 0.0)
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(metric_names, metric_values, color=colors, width=0.55, edgecolor="black", linewidth=1.2)

    for bar, val in zip(bars, metric_values):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.02,
            f"{val:.2%}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11
        )

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=11, fontweight="bold")
    ax.set_title("NOMA User Grouping Performance Summary", fontsize=13, fontweight="bold", pad=15)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[VISUALIZATION] Saved metrics summary bar chart to: {save_path}")


def plot_label_efficiency(results: dict, save_dir: str = None):
    """
    Renders Grouping Accuracy, Precision, Recall, and F1-Score vs Labeled Data Percentage (5%, 10%, 20%, 30%, 50%).
    """
    if save_dir is None:
        save_dir = os.path.join(config.RESULTS_DIR, "plots")
    os.makedirs(save_dir, exist_ok=True)

    pcts = sorted(list(results.keys()))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    if len(pcts) > 0 and isinstance(results[pcts[0]], dict):
        accs = [results[p]["accuracy"] for p in pcts]
        precs = [results[p]["precision"] for p in pcts]
        recs = [results[p]["recall"] for p in pcts]
        f1s = [results[p]["f1_score"] for p in pcts]

        ax.plot(pcts, accs, 'o-', color='navy', linewidth=2.2, markersize=7, label="Accuracy")
        ax.plot(pcts, precs, 's--', color='crimson', linewidth=2.2, markersize=7, label="Precision (Macro)")
        ax.plot(pcts, recs, '^-.', color='darkgreen', linewidth=2.2, markersize=7, label="Recall (Macro)")
        ax.plot(pcts, f1s, 'd:', color='purple', linewidth=2.2, markersize=7, label="F1-Score (Macro)")

        for x, y in zip(pcts, accs):
            ax.annotate(f"{y:.1%}", (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontweight='bold', color='navy', fontsize=8.5)
        for x, y in zip(pcts, f1s):
            ax.annotate(f"{y:.1%}", (x, y), textcoords="offset points", xytext=(0, -15), ha='center', fontweight='bold', color='purple', fontsize=8.5)

        ax.set_title("Label Efficiency: Accuracy, Precision, Recall & F1 vs Labeled Ratio", fontsize=12, fontweight='bold')
        ax.set_ylabel("Metric Score")
    else:
        accs = [results[p] for p in pcts]
        ax.plot(pcts, accs, 'o-', color='navy', linewidth=2.5, markersize=8, label="Proposed GNN")
        for x, y in zip(pcts, accs):
            ax.annotate(f"{y:.2%}", (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold')
        ax.set_title("Label Efficiency: GNN Grouping Accuracy vs Labeled Ratio", fontsize=12, fontweight='bold')
        ax.set_ylabel("Test Grouping Accuracy")

    ax.set_xlabel("Labeled Data Percentage (%)", fontweight='bold')
    ax.set_ylim(0.0, 1.15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    plt.tight_layout()
    save_path = os.path.join(save_dir, "label_efficiency.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[VISUALIZATION] Saved label efficiency plot to: {save_path}")


def plot_csi_uncertainty_impact(binned_df: pd.DataFrame, save_dir: str = None):
    """
    Renders:
    1. CSI Uncertainty vs GNN Grouping Accuracy, Precision, Recall & F1
    2. CSI Uncertainty vs Prediction Confidence
    """
    if save_dir is None:
        save_dir = os.path.join(config.RESULTS_DIR, "plots")
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    # Plot 1: Accuracy, Precision, Recall, F1 vs CSI Uncertainty
    ax1.plot(binned_df["mean_csi_uncertainty"], binned_df["grouping_accuracy"], 'ro-', linewidth=2, markersize=7, label="Accuracy")
    if "grouping_precision" in binned_df.columns:
        ax1.plot(binned_df["mean_csi_uncertainty"], binned_df["grouping_precision"], 'md--', linewidth=2, markersize=7, label="Precision")
    if "grouping_recall" in binned_df.columns:
        ax1.plot(binned_df["mean_csi_uncertainty"], binned_df["grouping_recall"], 'g^-.', linewidth=2, markersize=7, label="Recall")
    if "grouping_f1" in binned_df.columns:
        ax1.plot(binned_df["mean_csi_uncertainty"], binned_df["grouping_f1"], 'cs:', linewidth=2, markersize=7, label="F1-Score")

    ax1.set_title("GNN Metrics vs CSI Uncertainty", fontweight='bold')
    ax1.set_xlabel("Mean CSI Uncertainty", fontweight='bold')
    ax1.set_ylabel("Score", fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Prediction Confidence vs CSI Uncertainty
    ax2.plot(binned_df["mean_csi_uncertainty"], binned_df["mean_confidence"], 'bs-', linewidth=2, markersize=7)
    ax2.set_title("Prediction Confidence vs CSI Uncertainty", fontweight='bold')
    ax2.set_xlabel("Mean CSI Uncertainty", fontweight='bold')
    ax2.set_ylabel("Mean Softmax Confidence", fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "csi_uncertainty_impact.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[VISUALIZATION] Saved CSI uncertainty impact plot to: {save_path}")


def plot_model_architecture_diagrams(save_dir: str = None):
    """
    Generates required model architecture diagrams (Section 32):
    1. System Pipeline Flowchart
    2. Semi-Supervised Dual-Perturbation Training Loop Flowchart
    """
    if save_dir is None:
        save_dir = os.path.join(config.RESULTS_DIR, "plots")
    os.makedirs(save_dir, exist_ok=True)

    # Diagram 1: System Pipeline
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis('off')
    pipeline_steps = [
        "External Wireless CSI Dataset",
        "Data Preprocessing & Scaler",
        "CSI Feature Extraction & Uncertainty Derivation",
        "Graph Builder (User Nodes & Similarity Edges)",
        f"GNN Model ({config.GNN_TYPE} Backbone)",
        "Node Embedding Representation",
        "Softmax Classification Head",
        "NOMA User Group Prediction Output"
    ]

    for i, step in enumerate(pipeline_steps):
        ax.text(
            0.5, 0.9 - i * 0.11, step,
            ha='center', va='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", edgecolor="darkcyan", linewidth=1.5)
        )
        if i < len(pipeline_steps) - 1:
            ax.annotate("", xy=(0.5, 0.84 - i * 0.11), xytext=(0.5, 0.88 - i * 0.11),
                        arrowprops=dict(arrowstyle="->", lw=2, color="darkcyan"))

    ax.set_title("Uncertainty-Aware GNN End-to-End System Pipeline", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "model_pipeline_architecture.png"), dpi=300)
    plt.close()

    # Diagram 2: Training Flowchart
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis('off')

    ax.text(0.25, 0.85, "Labeled Graph\n(Known Groups)", ha='center', va='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))
    ax.text(0.75, 0.85, "Unlabeled Graph\n(Unknown Groups)", ha='center', va='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightpink"))

    ax.annotate("", xy=(0.25, 0.68), xytext=(0.25, 0.78), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(0.25, 0.65, "GNN\nForward", ha='center', va='center', fontsize=9, bbox=dict(boxstyle="square,pad=0.3", facecolor="white"))

    ax.annotate("", xy=(0.75, 0.72), xytext=(0.75, 0.78), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(0.75, 0.70, "Graph Perturbations\n(Edge Drop / Feature Mask)", ha='center', va='center', fontsize=9, bbox=dict(boxstyle="square,pad=0.3", facecolor="white"))

    ax.text(0.60, 0.55, "Graph A", ha='center', va='center', fontsize=9, bbox=dict(boxstyle="circle,pad=0.3", facecolor="yellow"))
    ax.text(0.90, 0.55, "Graph B", ha='center', va='center', fontsize=9, bbox=dict(boxstyle="circle,pad=0.3", facecolor="yellow"))

    ax.annotate("", xy=(0.60, 0.60), xytext=(0.70, 0.65), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(0.90, 0.60), xytext=(0.80, 0.65), arrowprops=dict(arrowstyle="->", lw=1.5))

    ax.text(0.25, 0.45, "Supervised Loss\n(Cross Entropy)", ha='center', va='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgreen"))
    ax.text(0.75, 0.45, "Consistency Loss (KL/MSE)\n+ Pseudo-Label Loss", ha='center', va='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.4", facecolor="plum"))

    ax.annotate("", xy=(0.25, 0.50), xytext=(0.25, 0.60), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(0.60, 0.49), xytext=(0.60, 0.52), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(0.90, 0.49), xytext=(0.90, 0.52), arrowprops=dict(arrowstyle="->", lw=1.5))

    ax.text(0.50, 0.25, "TOTAL LOSS\nL = L_sup + λ L_cons + μ L_pseudo", ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.6", facecolor="gold", edgecolor="darkgoldenrod", lw=2))

    ax.annotate("", xy=(0.50, 0.31), xytext=(0.25, 0.40), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(0.50, 0.31), xytext=(0.75, 0.40), arrowprops=dict(arrowstyle="->", lw=2))

    ax.text(0.50, 0.08, "GNN Gradient Backpropagation & Parameter Update", ha='center', va='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.4", facecolor="orange", edgecolor="darkorange", lw=1.5))

    ax.annotate("", xy=(0.50, 0.12), xytext=(0.50, 0.19), arrowprops=dict(arrowstyle="->", lw=2))

    ax.set_title("Semi-Supervised Uncertainty-Aware GNN Dual-Perturbation Training Loop", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "semi_supervised_training_architecture.png"), dpi=300)
    plt.close()

    print(f"[VISUALIZATION] Saved model architecture flowcharts to: {save_dir}")
