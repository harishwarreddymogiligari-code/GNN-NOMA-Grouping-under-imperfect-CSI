import os
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any

import config
from models import UncertaintyAwareGNN
from evaluation.metrics import calculate_classification_metrics
from preprocessing.graph_builder import build_scenario_graph


def run_baseline_comparison(
    train_graphs: List[torch.nn.Module],
    val_graphs: List[torch.nn.Module],
    test_graphs: List[torch.nn.Module],
    in_channels: int,
    device: str = "cpu"
) -> Dict[str, Dict[str, float]]:
    """
    Executes comparison baseline models required by Section 22:
    - Model 1: Fully Supervised GNN
    - Model 2: Semi-Supervised GNN (without CSI Uncertainty feature)
    - Model 3: Proposed Semi-Supervised Uncertainty-Aware GNN
    """
    print("\n========================================")
    print("RUNNING BASELINE MODEL COMPARISON SUITE")
    print("========================================")
    from training.trainer import SemiSupervisedTrainer
    results = {}

    # ----------------------------------------------------
    # Model 1: Fully Supervised GNN (lambda=0, mu=0)
    # ----------------------------------------------------
    print("[BASELINE 1] Training Fully Supervised GNN (No Consistency/Pseudo Loss)...")
    m1 = UncertaintyAwareGNN(in_channels, config.HIDDEN_DIM, config.NUM_GROUPS, config.GNN_TYPE, config.NUM_LAYERS, config.DROPOUT)
    t1 = SemiSupervisedTrainer(m1, lambda_consistency=0.0, mu_pseudo=0.0, device=device)
    t1.fit(train_graphs, val_graphs, epochs=config.EPOCHS // 2, model_save_dir=os.path.join(config.MODEL_SAVE_DIR, "b1"))
    m1_metrics = t1.evaluate(test_graphs)
    results["Fully Supervised GNN"] = m1_metrics
    print(f" -> Model 1 Test Accuracy: {m1_metrics['accuracy']:.4f} | F1: {m1_metrics['f1_score']:.4f}")

    # ----------------------------------------------------
    # Model 2: Semi-Supervised GNN (Without CSI Uncertainty)
    # ----------------------------------------------------
    print("[BASELINE 2] Training Semi-Supervised GNN (Without CSI Uncertainty feature)...")
    # Clone graphs and mask out CSI uncertainty feature index (index 2 in config.FEATURE_COLUMNS)
    uncert_idx = config.FEATURE_COLUMNS.index("csi_uncertainty") if "csi_uncertainty" in config.FEATURE_COLUMNS else 2

    def remove_uncert_feat(graph_list):
        cloned = []
        for g in graph_list:
            g_c = g.clone()
            g_c.x = g_c.x.clone()
            g_c.x[:, uncert_idx] = 0.0  # Zero out CSI uncertainty
            cloned.append(g_c)
        return cloned

    train_no_uncert = remove_uncert_feat(train_graphs)
    val_no_uncert = remove_uncert_feat(val_graphs)
    test_no_uncert = remove_uncert_feat(test_graphs)

    m2 = UncertaintyAwareGNN(in_channels, config.HIDDEN_DIM, config.NUM_GROUPS, config.GNN_TYPE, config.NUM_LAYERS, config.DROPOUT)
    t2 = SemiSupervisedTrainer(m2, lambda_consistency=config.LAMBDA_CONSISTENCY, mu_pseudo=config.MU_PSEUDO, device=device)
    t2.fit(train_no_uncert, val_no_uncert, epochs=config.EPOCHS // 2, model_save_dir=os.path.join(config.MODEL_SAVE_DIR, "b2"))
    m2_metrics = t2.evaluate(test_no_uncert)
    results["Semi-Supervised (No CSI Uncertainty)"] = m2_metrics
    print(f" -> Model 2 Test Accuracy: {m2_metrics['accuracy']:.4f} | F1: {m2_metrics['f1_score']:.4f}")

    # ----------------------------------------------------
    # Model 3: Proposed Semi-Supervised Uncertainty-Aware GNN
    # ----------------------------------------------------
    print("[PROPOSED MODEL 3] Training Proposed Uncertainty-Aware Semi-Supervised GNN...")
    m3 = UncertaintyAwareGNN(in_channels, config.HIDDEN_DIM, config.NUM_GROUPS, config.GNN_TYPE, config.NUM_LAYERS, config.DROPOUT)
    t3 = SemiSupervisedTrainer(m3, lambda_consistency=config.LAMBDA_CONSISTENCY, mu_pseudo=config.MU_PSEUDO, device=device)
    t3.fit(train_graphs, val_graphs, epochs=config.EPOCHS, model_save_dir=config.MODEL_SAVE_DIR)
    m3_metrics = t3.evaluate(test_graphs)
    results["Proposed Uncertainty-Aware Semi-Supervised GNN"] = m3_metrics
    print(f" -> Model 3 Test Accuracy: {m3_metrics['accuracy']:.4f} | F1: {m3_metrics['f1_score']:.4f}")

    return results


def run_label_efficiency_experiment(
    full_train_df: pd.DataFrame,
    val_graphs: List[torch.nn.Module],
    test_graphs: List[torch.nn.Module],
    scaler: Any,
    in_channels: int,
    percentages: List[float] = [5.0, 10.0, 20.0, 30.0, 50.0],
    device: str = "cpu"
) -> Dict[float, float]:
    """
    Executes Label Efficiency Experiment required by Section 20:
    Trains model with 5%, 10%, 20%, 30%, 50% labeled training split ratios
    and measures grouping accuracy on test dataset.
    """
    print("\n========================================")
    print("RUNNING LABEL EFFICIENCY EXPERIMENT")
    print("========================================")
    from training.trainer import SemiSupervisedTrainer
    accuracy_results = {}

    for p in percentages:
        print(f"\n[LABEL EFFICIENCY] Training with {p}% labeled data...")
        # Create shallow copy of training dataframe
        df_p = full_train_df.copy()

        # Mask labels to retain only p% of true labels
        labels = df_p[config.LABEL_COLUMN].values.copy()
        n_samples = len(labels)
        n_labeled = int(n_samples * (p / 100.0))

        # Randomly choose indices to keep labeled
        np.random.seed(config.RANDOM_SEED)
        unlabeled_indices = np.random.choice(n_samples, size=n_samples - n_labeled, replace=False)
        labels[unlabeled_indices] = -1
        df_p[config.LABEL_COLUMN] = labels

        # Transform features
        df_p_scaled = scaler.transform(df_p)

        # Build PyG training graphs
        from preprocessing.graph_builder import build_pyg_datasets
        p_train_graphs = build_pyg_datasets(df_p_scaled, config.FEATURE_COLUMNS, config.LABEL_COLUMN, config.GRAPH_METHOD, config.K_NEIGHBORS)

        # Train model
        model = UncertaintyAwareGNN(in_channels, config.HIDDEN_DIM, config.NUM_GROUPS, config.GNN_TYPE, config.NUM_LAYERS, config.DROPOUT)
        trainer = SemiSupervisedTrainer(model, device=device)
        trainer.fit(p_train_graphs, val_graphs, epochs=config.EPOCHS // 2, model_save_dir=os.path.join(config.MODEL_SAVE_DIR, f"lbl_{int(p)}"))

        # Evaluate on test set
        test_metrics = trainer.evaluate(test_graphs)
        accuracy_results[p] = test_metrics["accuracy"]
        print(f" -> Labeled Split {p}%: Test Grouping Accuracy = {test_metrics['accuracy']:.4f}")

    return accuracy_results


def run_csi_uncertainty_experiment(
    model: torch.nn.Module,
    test_graphs: List[torch.nn.Module],
    device: str = "cpu"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes CSI Uncertainty Sensitivity Experiment required by Section 21:
    Evaluates GNN grouping accuracy and prediction confidence across different
    CSI uncertainty levels.
    """
    print("\n========================================")
    print("RUNNING CSI UNCERTAINTY SENSITIVITY EXPERIMENT")
    print("========================================")
    model.eval()
    model.to(device)

    all_uncert = []
    all_conf = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for graph in test_graphs:
            graph = graph.to(device)
            probs, _, _ = model(graph.x, graph.edge_index, graph.edge_attr)
            max_probs, preds = torch.max(probs, dim=-1)

            all_uncert.extend(graph.csi_uncertainty.cpu().numpy())
            all_conf.extend(max_probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(graph.y.cpu().numpy())

    df_res = pd.DataFrame({
        "csi_uncertainty": all_uncert,
        "confidence": all_conf,
        "pred": all_preds,
        "target": all_targets
    })

    # Keep only labeled test samples for accuracy evaluation
    df_valid = df_res[df_res["target"] != -1].copy()
    df_valid["correct"] = (df_valid["pred"] == df_valid["target"]).astype(float)

    # Bin into uncertainty quantiles (e.g., 5 bins)
    try:
        df_valid["uncert_bin"] = pd.qcut(df_valid["csi_uncertainty"], q=5, duplicates='drop')
    except Exception:
        df_valid["uncert_bin"] = pd.cut(df_valid["csi_uncertainty"], bins=5)

    binned_metrics = df_valid.groupby("uncert_bin", observed=False).agg(
        mean_csi_uncertainty=("csi_uncertainty", "mean"),
        grouping_accuracy=("correct", "mean"),
        mean_confidence=("confidence", "mean"),
        sample_count=("correct", "count")
    ).reset_index()

    print(binned_metrics.to_string(index=False))

    return df_res, binned_metrics
