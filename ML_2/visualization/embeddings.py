import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import config


def plot_node_embeddings(
    embeddings: np.ndarray,
    predicted_groups: np.ndarray,
    true_groups: np.ndarray = None,
    save_dir: str = None
):
    """
    Applies PCA and t-SNE dimensionality reduction on learned GNN node embeddings
    and generates 2D scatter plots colored by predicted NOMA groups and true labels.
    """
    if save_dir is None:
        save_dir = os.path.join(config.RESULTS_DIR, "embeddings")
    os.makedirs(save_dir, exist_ok=True)

    if len(embeddings) < 2:
        print("[EMBEDDINGS] Insufficient embedding samples for 2D plot.")
        return

    # PCA 2D reduction
    pca = PCA(n_components=2, random_state=config.RANDOM_SEED)
    emb_pca = pca.fit_transform(embeddings)

    # t-SNE 2D reduction (perplexity adjusted for small sample size)
    perp = min(30, max(5, len(embeddings) - 1))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=config.RANDOM_SEED)
    emb_tsne = tsne.fit_transform(embeddings)

    # 1. Plot PCA by Predicted Group
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        x=emb_pca[:, 0], y=emb_pca[:, 1],
        hue=[f"Group {g+1}" for g in predicted_groups],
        palette="Set1", s=70, alpha=0.8, ax=ax
    )
    ax.set_title("GNN Node Embeddings (PCA 2D) - Colored by Predicted Group", fontsize=12, fontweight='bold')
    ax.set_xlabel("PCA Dimension 1")
    ax.set_ylabel("PCA Dimension 2")
    plt.tight_layout()
    pca_path = os.path.join(save_dir, "gnn_embeddings_pca.png")
    plt.savefig(pca_path, dpi=300)
    plt.close()

    # 2. Plot t-SNE by Predicted Group
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        x=emb_tsne[:, 0], y=emb_tsne[:, 1],
        hue=[f"Group {g+1}" for g in predicted_groups],
        palette="Set1", s=70, alpha=0.8, ax=ax
    )
    ax.set_title("GNN Node Embeddings (t-SNE 2D) - Colored by Predicted Group", fontsize=12, fontweight='bold')
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    tsne_path = os.path.join(save_dir, "gnn_embeddings_tsne.png")
    plt.savefig(tsne_path, dpi=300)
    plt.close()

    # 3. Plot PCA/t-SNE by True Group if available
    if true_groups is not None:
        valid_mask = (true_groups != -1)
        if valid_mask.any():
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(
                x=emb_pca[valid_mask, 0], y=emb_pca[valid_mask, 1],
                hue=[f"True Group {g+1}" for g in true_groups[valid_mask]],
                palette="Set2", s=70, alpha=0.8, ax=ax
            )
            ax.set_title("GNN Node Embeddings (PCA 2D) - Colored by Ground Truth Group", fontsize=12, fontweight='bold')
            ax.set_xlabel("PCA Dimension 1")
            ax.set_ylabel("PCA Dimension 2")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "gnn_embeddings_pca_true_labels.png"), dpi=300)
            plt.close()

    print(f"[VISUALIZATION] Saved embedding scatter plots to: {save_dir}")
