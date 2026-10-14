import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import numpy as np
import pandas as pd
import argparse

import config
from data.generate_demo_data import generate_noma_dataset
from preprocessing import load_raw_dataset, print_dataset_info, FeatureScaler, build_pyg_datasets
from models import UncertaintyAwareGNN
from training import SemiSupervisedTrainer
from evaluation import (
    calculate_classification_metrics,
    run_baseline_comparison,
    run_label_efficiency_experiment,
    run_csi_uncertainty_experiment
)
from visualization import (
    plot_noma_user_graph,
    save_graphml_file,
    plot_node_embeddings,
    plot_training_history,
    plot_label_efficiency,
    plot_csi_uncertainty_impact,
    plot_model_architecture_diagrams
)
from visualization.graph_visualization import build_networkx_graph


def main():
    parser = argparse.ArgumentParser(description="Train Uncertainty-Aware GNN for Dynamic NOMA User Grouping")
    parser.add_argument("--dataset", type=str, default=config.DATASET_PATH, help="Path to input external CSV dataset")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of training epochs")
    parser.add_argument("--labeled_pct", type=float, default=config.LABELED_PERCENTAGE, help="Percentage of labeled training data")
    args = parser.parse_args()

    # Ensure output directories exist
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(os.path.join(config.RESULTS_DIR, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(config.RESULTS_DIR, "graphs"), exist_ok=True)
    os.makedirs(os.path.join(config.RESULTS_DIR, "embeddings"), exist_ok=True)
    os.makedirs(os.path.join(config.RESULTS_DIR, "plots"), exist_ok=True)

    # 1. Dataset Loading / Generation
    if not os.path.exists(args.dataset):
        print(f"[TRAIN] Dataset not found at '{args.dataset}'. Generating realistic demo dataset...")
        os.makedirs(os.path.dirname(args.dataset), exist_ok=True)
        df_demo = generate_noma_dataset(num_scenarios=100, users_per_scenario=20, num_groups=config.NUM_GROUPS)
        df_demo.to_csv(args.dataset, index=False)

    df_raw = load_raw_dataset(args.dataset)

    # 2. Scenario-based Data Splitting (70% Train, 15% Val, 15% Test)
    np.random.seed(config.RANDOM_SEED)
    scenarios = df_raw[config.SCENARIO_COLUMN].unique()
    np.random.shuffle(scenarios)

    n_scenarios = len(scenarios)
    n_train = int(n_scenarios * config.TRAIN_RATIO)
    n_val = int(n_scenarios * config.VAL_RATIO)

    train_scenarios = scenarios[:n_train]
    val_scenarios = scenarios[n_train:n_train + n_val]
    test_scenarios = scenarios[n_train + n_val:]

    df_train = df_raw[df_raw[config.SCENARIO_COLUMN].isin(train_scenarios)].copy()
    df_val = df_raw[df_raw[config.SCENARIO_COLUMN].isin(val_scenarios)].copy()
    df_test = df_raw[df_raw[config.SCENARIO_COLUMN].isin(test_scenarios)].copy()

    # 3. Apply Semi-Supervised Label Masking on Training Set (Case B/C)
    full_train_df = df_train.copy()  # Save copy with true labels for efficiency experiments
    n_train_samples = len(df_train)
    n_labeled = int(n_train_samples * (args.labeled_pct / 100.0))

    unlabeled_indices = np.random.choice(n_train_samples, size=n_train_samples - n_labeled, replace=False)
    train_labels = df_train[config.LABEL_COLUMN].values.copy()
    train_labels[unlabeled_indices] = -1
    df_train[config.LABEL_COLUMN] = train_labels

    # Display dataset diagnostic summary
    print_dataset_info(df_raw, num_labeled=n_labeled, num_unlabeled=(len(df_raw) - n_labeled))

    # 4. Feature Normalization (Fit Scaler strictly on Training Data to prevent leakage)
    scaler = FeatureScaler(config.FEATURE_COLUMNS)
    scaler.fit(df_train)

    df_train_scaled = scaler.transform(df_train)
    df_val_scaled = scaler.transform(df_val)
    df_test_scaled = scaler.transform(df_test)

    # 5. Build PyTorch Geometric Graph Datasets
    print(f"\n[GRAPH BUILDER] Constructing user graphs using method: '{config.GRAPH_METHOD}' (k={config.K_NEIGHBORS})...")
    train_graphs = build_pyg_datasets(df_train_scaled, config.FEATURE_COLUMNS, config.LABEL_COLUMN, config.GRAPH_METHOD, config.K_NEIGHBORS)
    val_graphs = build_pyg_datasets(df_val_scaled, config.FEATURE_COLUMNS, config.LABEL_COLUMN, config.GRAPH_METHOD, config.K_NEIGHBORS)
    test_graphs = build_pyg_datasets(df_test_scaled, config.FEATURE_COLUMNS, config.LABEL_COLUMN, config.GRAPH_METHOD, config.K_NEIGHBORS)

    in_channels = len(config.FEATURE_COLUMNS)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[MODEL] Initializing Uncertainty-Aware GNN ({config.GNN_TYPE}) on device: {device}...")

    # 6. Model Training
    model = UncertaintyAwareGNN(
        in_channels=in_channels,
        hidden_dim=config.HIDDEN_DIM,
        out_channels=config.NUM_GROUPS,
        gnn_type=config.GNN_TYPE,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT
    )

    trainer = SemiSupervisedTrainer(
        model=model,
        learning_rate=config.LEARNING_RATE,
        lambda_consistency=config.LAMBDA_CONSISTENCY,
        mu_pseudo=config.MU_PSEUDO,
        confidence_threshold=config.CONFIDENCE_THRESHOLD,
        uncertainty_threshold=config.UNCERTAINTY_THRESHOLD,
        device=device
    )

    history = trainer.fit(train_graphs, val_graphs, epochs=args.epochs, model_save_dir=config.MODEL_SAVE_DIR)

    # Load best saved checkpoint for inference & evaluation
    best_checkpoint = os.path.join(config.MODEL_SAVE_DIR, "best_semi_supervised_gnn.pt")
    model.load_state_dict(torch.load(best_checkpoint, map_location=device))
    model.eval()
    model.to(device)

    # 7. Model Inference on Test Set
    print("\n========================================")
    print("TEST SET EVALUATION & PREDICTION EXPORT")
    print("========================================")
    all_user_ids = []
    all_preds = []
    all_confs = []
    all_uncerts = []
    all_true_groups = []
    all_probs = []
    all_embeddings = []
    all_edges_src = []
    all_edges_dst = []
    all_edges_w = []

    with torch.no_grad():
        for graph in test_graphs:
            graph = graph.to(device)
            probs, embeddings, logits = model(graph.x, graph.edge_index, graph.edge_attr)
            max_probs, preds = torch.max(probs, dim=-1)

            all_user_ids.extend(graph.user_id.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_confs.extend(max_probs.cpu().numpy())
            all_uncerts.extend(graph.csi_uncertainty.cpu().numpy())
            all_true_groups.extend(graph.y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_embeddings.append(embeddings.cpu().numpy())

            # Export graph edge attributes
            e_idx = graph.edge_index.cpu().numpy()
            e_w = graph.edge_attr.cpu().numpy() if graph.edge_attr is not None else np.ones((e_idx.shape[1], 1))
            for i in range(e_idx.shape[1]):
                all_edges_src.append(int(graph.user_id[e_idx[0, i]].item()))
                all_edges_dst.append(int(graph.user_id[e_idx[1, i]].item()))
                all_edges_w.append(float(e_w[i, 0]))

    all_user_ids = np.array(all_user_ids)
    all_preds = np.array(all_preds)
    all_confs = np.array(all_confs)
    all_uncerts = np.array(all_uncerts)
    all_true_groups = np.array(all_true_groups)
    stacked_probs = np.vstack(all_probs)
    stacked_embeddings = np.vstack(all_embeddings)

    # Calculate Test Classification Metrics
    test_metrics = calculate_classification_metrics(all_true_groups, all_preds)
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Precision (Macro): {test_metrics['precision']:.4f}")
    print(f"Test Recall (Macro): {test_metrics['recall']:.4f}")
    print(f"Test F1-Score (Macro): {test_metrics['f1_score']:.4f}")
    print("\nConfusion Matrix:")
    print(test_metrics['confusion_matrix'])

    # 8. Save Required Output Files (Section 17 & 18)
    # 1. predicted_groups.csv
    pred_df = pd.DataFrame({
        "user_id": all_user_ids,
        "predicted_group": all_preds + 1,  # 1-indexed group labels
        "confidence": np.round(all_confs, 4),
        "csi_uncertainty": np.round(all_uncerts, 4)
    })
    pred_path = os.path.join(config.RESULTS_DIR, "predictions", "predicted_groups.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"\nSaved predicted user groups: {pred_path}")

    # 2. prediction_probabilities.csv
    prob_cols = [f"group_{i+1}_prob" for i in range(config.NUM_GROUPS)]
    prob_df = pd.DataFrame(stacked_probs, columns=prob_cols)
    prob_df.insert(0, "user_id", all_user_ids)
    prob_path = os.path.join(config.RESULTS_DIR, "predictions", "prediction_probabilities.csv")
    prob_df.to_csv(prob_path, index=False)
    print(f"Saved prediction probabilities matrix: {prob_path}")

    # 3. node_embeddings.csv
    emb_cols = [f"emb_dim_{i+1}" for i in range(stacked_embeddings.shape[1])]
    emb_df = pd.DataFrame(stacked_embeddings, columns=emb_cols)
    emb_df.insert(0, "user_id", all_user_ids)
    emb_path = os.path.join(config.RESULTS_DIR, "predictions", "node_embeddings.csv")
    emb_df.to_csv(emb_path, index=False)
    print(f"Saved GNN node embeddings: {emb_path}")

    # 4. graph_edges.csv
    edges_df = pd.DataFrame({
        "source_user_id": all_edges_src,
        "target_user_id": all_edges_dst,
        "edge_weight": all_edges_w
    })
    edges_path = os.path.join(config.RESULTS_DIR, "predictions", "graph_edges.csv")
    edges_df.to_csv(edges_path, index=False)
    print(f"Saved graph edge structure: {edges_path}")

    # 5. GraphML export and NetworkX visualization
    first_test_graph = test_graphs[0]
    first_preds = all_preds[:first_test_graph.num_nodes]
    first_confs = all_confs[:first_test_graph.num_nodes]
    first_uncerts = all_uncerts[:first_test_graph.num_nodes]

    nx_graph = build_networkx_graph(first_test_graph, first_preds, first_confs, first_uncerts)
    graphml_path = os.path.join(config.RESULTS_DIR, "graphs", "predicted_graph.graphml")
    save_graphml_file(nx_graph, graphml_path)
    plot_noma_user_graph(nx_graph)

    # 9. Generate Embeddings & Plot Visualizations
    plot_node_embeddings(stacked_embeddings, all_preds, all_true_groups)
    plot_training_history(history)
    plot_model_architecture_diagrams()

    # 10. Run Critical Experiments
    # CSI Uncertainty Sensitivity Experiment
    _, binned_metrics = run_csi_uncertainty_experiment(model, test_graphs, device)
    plot_csi_uncertainty_impact(binned_metrics)

    # Label Efficiency Experiment (5%, 10%, 20%, 30%, 50%)
    label_eff_results = run_label_efficiency_experiment(full_train_df, val_graphs, test_graphs, scaler, in_channels, device=device)
    plot_label_efficiency(label_eff_results)

    # Baseline Model Comparisons
    baseline_results = run_baseline_comparison(train_graphs, val_graphs, test_graphs, in_channels, device)

    print("\n========================================")
    print("ALL TRAINING & EVALUATION TASKS COMPLETED SUCCESSFULLY!")
    print("========================================")


if __name__ == "__main__":
    main()
