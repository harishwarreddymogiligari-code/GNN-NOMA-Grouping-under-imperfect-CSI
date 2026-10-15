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
    print(f" -> Model 1 Test Accuracy: {m1_metrics['accuracy']:.4f} | Precision: {m1_metrics['precision']:.4f} | F1: {m1_metrics['f1_score']:.4f}")

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
    print(f" -> Model 2 Test Accuracy: {m2_metrics['accuracy']:.4f} | Precision: {m2_metrics['precision']:.4f} | F1: {m2_metrics['f1_score']:.4f}")

    # ----------------------------------------------------
    # Model 3: Proposed Semi-Supervised Uncertainty-Aware GNN
    # ----------------------------------------------------
    print("[PROPOSED MODEL 3] Training Proposed Uncertainty-Aware Semi-Supervised GNN...")
    m3 = UncertaintyAwareGNN(in_channels, config.HIDDEN_DIM, config.NUM_GROUPS, config.GNN_TYPE, config.NUM_LAYERS, config.DROPOUT)
    t3 = SemiSupervisedTrainer(m3, lambda_consistency=config.LAMBDA_CONSISTENCY, mu_pseudo=config.MU_PSEUDO, device=device)
    t3.fit(train_graphs, val_graphs, epochs=config.EPOCHS, model_save_dir=config.MODEL_SAVE_DIR)
    m3_metrics = t3.evaluate(test_graphs)
    results["Proposed Uncertainty-Aware Semi-Supervised GNN"] = m3_metrics
    print(f" -> Model 3 Test Accuracy: {m3_metrics['accuracy']:.4f} | Precision: {m3_metrics['precision']:.4f} | F1: {m3_metrics['f1_score']:.4f}")

    # Save baseline comparison summary CSV
    baseline_records = []
    for model_name, m in results.items():
        baseline_records.append({
            "model": model_name,
            "accuracy": round(m["accuracy"], 4),
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "f1_score": round(m["f1_score"], 4)
        })
    os.makedirs(os.path.join(config.RESULTS_DIR, "predictions"), exist_ok=True)
    baseline_df = pd.DataFrame(baseline_records)
    baseline_df.to_csv(os.path.join(config.RESULTS_DIR, "predictions", "baseline_comparison.csv"), index=False)

    return results


def run_label_efficiency_experiment(
    full_train_df: pd.DataFrame,
    val_graphs: List[torch.nn.Module],
    test_graphs: List[torch.nn.Module],
    scaler: Any,
    in_channels: int,
    percentages: List[float] = [5.0, 10.0, 20.0, 30.0, 50.0],
    device: str = "cpu"
) -> Dict[float, Dict[str, float]]:
    """
    Executes Label Efficiency Experiment required by Section 20:
    Trains model with 5%, 10%, 20%, 30%, 50% labeled training split ratios
    and measures grouping accuracy & precision on test dataset.
    """
    print("\n========================================")
    print("RUNNING LABEL EFFICIENCY EXPERIMENT")
    print("========================================")
    from training.trainer import SemiSupervisedTrainer
    efficiency_results = {}

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
        efficiency_results[p] = {
            "accuracy": test_metrics["accuracy"],
            "precision": test_metrics["precision"]
        }
        print(f" -> Labeled Split {p}%: Accuracy = {test_metrics['accuracy']:.4f} | Precision = {test_metrics['precision']:.4f}")

    # Save label efficiency CSV
    eff_records = [
        {"labeled_percentage": p, "accuracy": round(res["accuracy"], 4), "precision": round(res["precision"], 4)}
        for p, res in efficiency_results.items()
    ]
    eff_df = pd.DataFrame(eff_records)
    eff_df.to_csv(os.path.join(config.RESULTS_DIR, "predictions", "label_efficiency.csv"), index=False)

    return efficiency_results


def run_csi_uncertainty_experiment(
    model: torch.nn.Module,
    test_graphs: List[torch.nn.Module],
    device: str = "cpu"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes CSI Uncertainty Sensitivity Experiment required by Section 21:
    Evaluates GNN grouping accuracy, precision, and prediction confidence across
    different CSI uncertainty levels.
    """
    print("\n========================================")
    print("RUNNING CSI UNCERTAINTY SENSITIVITY EXPERIMENT")
    print("========================================")
    from sklearn.metrics import precision_score
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

    def compute_bin_metrics(g):
        acc = float((g["pred"] == g["target"]).mean())
        prec = float(precision_score(g["target"], g["pred"], average='macro', zero_division=0))
        return pd.Series({
            "mean_csi_uncertainty": float(g["csi_uncertainty"].mean()),
            "grouping_accuracy": round(acc, 4),
            "grouping_precision": round(prec, 4),
            "mean_confidence": round(float(g["confidence"].mean()), 4),
            "sample_count": int(len(g))
        })

    binned_metrics = df_valid.groupby("uncert_bin", observed=False).apply(compute_bin_metrics).reset_index()
    print(binned_metrics.to_string(index=False))

    # Save csi uncertainty metrics CSV
    binned_metrics.to_csv(os.path.join(config.RESULTS_DIR, "predictions", "csi_uncertainty_metrics.csv"), index=False)

    return df_res, binned_metrics
