import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data
from typing import Optional
import config


def build_networkx_graph(
    pyg_data: Data,
    predicted_groups: np.ndarray,
    confidences: np.ndarray,
    csi_uncertainties: np.ndarray
) -> nx.Graph:
    """
    Constructs a NetworkX graph with node and edge attributes derived from PyG Data and GNN predictions.
    """
    G = nx.Graph()

    num_nodes = pyg_data.num_nodes
    user_ids = pyg_data.user_id.cpu().numpy() if hasattr(pyg_data, 'user_id') else np.arange(1, num_nodes + 1)

    # Add nodes with metadata
    for i in range(num_nodes):
        uid = int(user_ids[i])
        pred_g = int(predicted_groups[i])
        conf = float(confidences[i])
        uncert = float(csi_uncertainties[i])

        G.add_node(
            i,
            user_id=uid,
            predicted_group=pred_g,
            confidence=conf,
            csi_uncertainty=uncert,
            label_name=f"U{uid} [G{pred_g+1}]"
        )

    # Add edges with weights
    edge_index = pyg_data.edge_index.cpu().numpy()
    edge_attr = pyg_data.edge_attr.cpu().numpy() if pyg_data.edge_attr is not None else None

    for idx in range(edge_index.shape[1]):
        src = int(edge_index[0, idx])
        dst = int(edge_index[1, idx])
        if src < dst:  # Undirected display
            w = float(edge_attr[idx, 0]) if edge_attr is not None else 1.0
            G.add_edge(src, dst, edge_weight=w)

    return G


def save_graphml_file(G: nx.Graph, filepath: str = None):
    """
    Exports NetworkX user graph to a reusable GraphML file.
    """
    if filepath is None:
        os.makedirs(os.path.join(config.RESULTS_DIR, "graphs"), exist_ok=True)
        filepath = os.path.join(config.RESULTS_DIR, "graphs", "predicted_graph.graphml")
    else:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    nx.write_graphml(G, filepath)
    print(f"[VISUALIZATION] Exported GraphML user graph to: {filepath}")


def plot_noma_user_graph(
    G: nx.Graph,
    save_path: str = None,
    title: str = "GNN Dynamic NOMA User Grouping under Imperfect CSI"
):
    """
    Renders dynamic NetworkX + Matplotlib graph visualization:
    - Node color = Predicted NOMA Group
    - Node size = CSI Uncertainty level
    - Edge thickness = Relationship / Channel similarity weight
    - Node label = User ID & Group
    """
    if save_path is None:
        os.makedirs(os.path.join(config.RESULTS_DIR, "graphs"), exist_ok=True)
        save_path = os.path.join(config.RESULTS_DIR, "graphs", "noma_user_group_graph.png")
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Spring layout positioning for nodes
    pos = nx.spring_layout(G, seed=config.RANDOM_SEED, k=0.5)

    # Colors for NOMA groups
    groups = [G.nodes[n]['predicted_group'] for n in G.nodes]
    palette = plt.colormaps['Set1'].resampled(config.NUM_GROUPS)

    # Node sizes scaled by CSI uncertainty
    uncerts = np.array([G.nodes[n]['csi_uncertainty'] for n in G.nodes])
    if uncerts.max() > uncerts.min():
        norm_uncert = (uncerts - uncerts.min()) / (uncerts.max() - uncerts.min() + 1e-8)
    else:
        norm_uncert = np.zeros_like(uncerts)
    node_sizes = 500 + norm_uncert * 1500  # Scale between 500 and 2000

    # Edge widths scaled by weight
    weights = [G.edges[e].get('edge_weight', 1.0) for e in G.edges]
    if len(weights) > 0 and max(weights) > min(weights):
        norm_weights = [(w - min(weights)) / (max(weights) - min(weights) + 1e-8) for w in weights]
        edge_widths = [0.8 + 2.5 * nw for nw in norm_weights]
    else:
        edge_widths = [1.5] * len(G.edges)

    # Draw Edges
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.4, edge_color='gray', ax=ax)

    # Draw Nodes
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_color=groups,
        cmap=palette,
        node_size=node_sizes,
        alpha=0.9,
        edgecolors='black',
        linewidths=1.5,
        ax=ax
    )

    # Draw Labels
    labels = {n: G.nodes[n]['label_name'] for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight='bold', ax=ax)

    # Legend & Descriptions
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Custom legend for node sizes and colors
    group_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette(i), markersize=10, label=f"NOMA Group {i+1}")
        for i in range(config.NUM_GROUPS)
    ]
    
    legend_legend = ax.legend(handles=group_handles, title="Predicted NOMA Groups", loc="upper left", frameon=True)
    ax.add_artist(legend_legend)

    # Sub-legend explaining node size
    fig.text(
        0.15, 0.02,
        "Visualization Legend:\n"
        "• Node Color = Predicted NOMA Group\n"
        "• Node Size = CSI Uncertainty (Larger = Higher Uncertainty)\n"
        "• Edge Thickness = Channel Similarity Weight",
        fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8)
    )

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[VISUALIZATION] Saved NOMA user graph plot to: {save_path}")
