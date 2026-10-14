import torch
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import config


def build_scenario_graph(
    df_scenario: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    method: str = "knn",
    k_neighbors: int = 4
) -> Data:
    """
    Converts a single wireless scenario dataframe into a PyTorch Geometric Data graph object.
    
    Nodes represent wireless users.
    Edges represent inter-user channel characteristics / spatial proximity.
    
    Supported edge construction methods:
    1. 'knn': k-Nearest Neighbors on channel characteristics / distance.
    2. 'channel_similarity': Gaussian RBF kernel on channel characteristics.
    3. 'fully_connected': Complete graph between all users in scenario.
    """
    num_users = len(df_scenario)
    if num_users == 0:
        raise ValueError("Cannot construct graph from empty scenario dataframe.")

    # Extract node features (x)
    node_features = df_scenario[feature_cols].values.astype(np.float32)
    x = torch.tensor(node_features, dtype=torch.float)

    # Extract user labels (y)
    if label_col in df_scenario.columns:
        labels = df_scenario[label_col].values.astype(np.int64)
    else:
        labels = np.full(num_users, -1, dtype=np.int64)
    y = torch.tensor(labels, dtype=torch.long)

    # Extract metadata attributes
    user_ids = torch.tensor(df_scenario[config.USER_ID_COLUMN].values, dtype=torch.long) \
        if config.USER_ID_COLUMN in df_scenario.columns else torch.arange(num_users)
    csi_uncertainties = torch.tensor(df_scenario["csi_uncertainty"].values, dtype=torch.float) \
        if "csi_uncertainty" in df_scenario.columns else torch.zeros(num_users, dtype=torch.float)
    scenario_id = df_scenario[config.SCENARIO_COLUMN].iloc[0] \
        if config.SCENARIO_COLUMN in df_scenario.columns else 0

    # Pairwise feature distance matrix for graph construction
    # Use channel gain, distance, and CSI features for similarity
    channel_features = df_scenario[["channel_gain", "distance"]].values if all(col in df_scenario.columns for col in ["channel_gain", "distance"]) else node_features
    
    # Standardize channel features for metric computation
    feat_norm = (channel_features - channel_features.mean(axis=0)) / (channel_features.std(axis=0) + 1e-8)
    diff = feat_norm[:, np.newaxis, :] - feat_norm[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))  # [num_users, num_users]

    edges_src = []
    edges_dst = []
    edge_weights = []

    if method == "knn":
        k = min(k_neighbors, num_users - 1)
        for i in range(num_users):
            # Sort indices by distance to user i
            sorted_indices = np.argsort(dist_matrix[i])
            # Exclude self-loop in top-k calculation
            neighbors = [idx for idx in sorted_indices if idx != i][:k]
            for neighbor in neighbors:
                edges_src.append(i)
                edges_dst.append(neighbor)
                # Similarity inverse to feature distance
                sim = 1.0 / (1.0 + dist_matrix[i, neighbor])
                edge_weights.append(sim)

    elif method == "channel_similarity":
        sigma = np.median(dist_matrix) + 1e-6
        sim_matrix = np.exp(-(dist_matrix**2) / (2 * sigma**2))
        threshold = 0.3
        for i in range(num_users):
            for j in range(num_users):
                if i != j and sim_matrix[i, j] >= threshold:
                    edges_src.append(i)
                    edges_dst.append(j)
                    edge_weights.append(sim_matrix[i, j])

    elif method == "fully_connected":
        for i in range(num_users):
            for j in range(num_users):
                if i != j:
                    edges_src.append(i)
                    edges_dst.append(j)
                    sim = 1.0 / (1.0 + dist_matrix[i, j])
                    edge_weights.append(sim)

    else:
        raise ValueError(f"Unknown graph construction method: '{method}'. Choose from 'knn', 'channel_similarity', 'fully_connected'.")

    # If no edges were formed (isolated nodes), add self-loops or nearest neighbor
    if len(edges_src) == 0:
        for i in range(num_users):
            edges_src.append(i)
            edges_dst.append((i + 1) % num_users)
            edge_weights.append(1.0)

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        user_id=user_ids,
        csi_uncertainty=csi_uncertainties,
        scenario_id=scenario_id,
        num_nodes=num_users
    )
    return data


def build_pyg_datasets(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    method: str = "knn",
    k_neighbors: int = 4
) -> List[Data]:
    """
    Converts a complete dataset containing multiple scenarios into a list of PyG Data graphs.
    """
    graphs = []
    if config.SCENARIO_COLUMN in df.columns:
        grouped = df.groupby(config.SCENARIO_COLUMN)
        for _, scenario_df in grouped:
            graph = build_scenario_graph(scenario_df, feature_cols, label_col, method, k_neighbors)
            graphs.append(graph)
    else:
        # Single graph scenario
        graph = build_scenario_graph(df, feature_cols, label_col, method, k_neighbors)
        graphs.append(graph)

    return graphs
