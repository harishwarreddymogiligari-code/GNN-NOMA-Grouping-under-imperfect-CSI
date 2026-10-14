import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import numpy as np
import pandas as pd
import argparse

import config
from preprocessing import load_raw_dataset, FeatureScaler, build_pyg_datasets
from models import UncertaintyAwareGNN
from visualization import plot_noma_user_graph, save_graphml_file
from visualization.graph_visualization import build_networkx_graph


def main():
    parser = argparse.ArgumentParser(description="Standalone Inference CLI for NOMA User Grouping GNN")
    parser.add_argument("--input", type=str, required=True, help="Path to input external dataset CSV file")
    parser.add_argument("--model", type=str, default=os.path.join(config.MODEL_SAVE_DIR, "best_semi_supervised_gnn.pt"), help="Path to trained PyTorch GNN checkpoint")
    parser.add_argument("--output", type=str, default=config.RESULTS_DIR, help="Directory to save output CSVs, GraphML, and plots")
    args = parser.parse_args()

    output_dir = args.output
    os.makedirs(os.path.join(output_dir, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "graphs"), exist_ok=True)

    print(f"\n========================================")
    print(f"RUNNING NOMA USER GROUPING INFERENCE")
    print(f"========================================")
    print(f"Input Dataset: {args.input}")
    print(f"Model Checkpoint: {args.model}")

    # 1. Load Raw External Dataset
    df_raw = load_raw_dataset(args.input)

    # 2. Feature Preprocessing / Normalization
    scaler = FeatureScaler(config.FEATURE_COLUMNS)
    scaler.fit(df_raw)  # In inference mode on unlabelled test data, scale features
    df_scaled = scaler.transform(df_raw)

    # 3. Graph Construction
    print(f"[PREDICT] Building user graphs using '{config.GRAPH_METHOD}' (k={config.K_NEIGHBORS})...")
    graphs = build_pyg_datasets(df_scaled, config.FEATURE_COLUMNS, config.LABEL_COLUMN, config.GRAPH_METHOD, config.K_NEIGHBORS)

    # 4. Load Trained GNN Model
    in_channels = len(config.FEATURE_COLUMNS)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UncertaintyAwareGNN(
        in_channels=in_channels,
        hidden_dim=config.HIDDEN_DIM,
        out_channels=config.NUM_GROUPS,
        gnn_type=config.GNN_TYPE,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT
    )

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Trained model checkpoint not found at: {args.model}. Please run 'python train.py' first.")

    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    model.to(device)

    # 5. Model Inference
    all_user_ids = []
    all_preds = []
    all_confs = []
    all_uncerts = []
    all_probs = []
    all_embeddings = []
    all_edges_src = []
    all_edges_dst = []
    all_edges_w = []

    with torch.no_grad():
        for graph in graphs:
            graph = graph.to(device)
            probs, embeddings, logits = model(graph.x, graph.edge_index, graph.edge_attr)
            max_probs, preds = torch.max(probs, dim=-1)

            all_user_ids.extend(graph.user_id.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_confs.extend(max_probs.cpu().numpy())
            all_uncerts.extend(graph.csi_uncertainty.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_embeddings.append(embeddings.cpu().numpy())

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
    stacked_probs = np.vstack(all_probs)
    stacked_embeddings = np.vstack(all_embeddings)

    # 6. Save Outputs (Section 17 & 18)
    # predicted_groups.csv
    pred_df = pd.DataFrame({
        "user_id": all_user_ids,
        "predicted_group": all_preds + 1,
        "confidence": np.round(all_confs, 4),
        "csi_uncertainty": np.round(all_uncerts, 4)
    })
    pred_path = os.path.join(output_dir, "predictions", "predicted_groups.csv")
    pred_df.to_csv(pred_path, index=False)

    # prediction_probabilities.csv
    prob_cols = [f"group_{i+1}_prob" for i in range(config.NUM_GROUPS)]
    prob_df = pd.DataFrame(stacked_probs, columns=prob_cols)
    prob_df.insert(0, "user_id", all_user_ids)
    prob_path = os.path.join(output_dir, "predictions", "prediction_probabilities.csv")
    prob_df.to_csv(prob_path, index=False)

    # node_embeddings.csv
    emb_cols = [f"emb_dim_{i+1}" for i in range(stacked_embeddings.shape[1])]
    emb_df = pd.DataFrame(stacked_embeddings, columns=emb_cols)
    emb_df.insert(0, "user_id", all_user_ids)
    emb_path = os.path.join(output_dir, "predictions", "node_embeddings.csv")
    emb_df.to_csv(emb_path, index=False)

    # graph_edges.csv
    edges_df = pd.DataFrame({
        "source_user_id": all_edges_src,
        "target_user_id": all_edges_dst,
        "edge_weight": all_edges_w
    })
    edges_path = os.path.join(output_dir, "predictions", "graph_edges.csv")
    edges_df.to_csv(edges_path, index=False)

    # NetworkX & GraphML Visualization
    first_graph = graphs[0]
    first_preds = all_preds[:first_graph.num_nodes]
    first_confs = all_confs[:first_graph.num_nodes]
    first_uncerts = all_uncerts[:first_graph.num_nodes]

    nx_graph = build_networkx_graph(first_graph, first_preds, first_confs, first_uncerts)
    graphml_path = os.path.join(output_dir, "graphs", "predicted_graph.graphml")
    save_graphml_file(nx_graph, graphml_path)
    
    graph_png_path = os.path.join(output_dir, "graphs", "predicted_noma_user_graph.png")
    plot_noma_user_graph(nx_graph, save_path=graph_png_path, title="Inference: Dynamic NOMA User Grouping")

    print("\n========================================")
    print("INFERENCE COMPLETED SUCCESSFULLY!")
    print(f"Results saved to directory: {output_dir}")
    print("========================================\n")


if __name__ == "__main__":
    main()
