# Uncertainty-Aware Graph Neural Network for Dynamic NOMA User Grouping under Imperfect CSI

A standalone **Python-only Semi-Supervised Graph Neural Network (GNN)** system for predicting **dynamic NOMA (Non-Orthogonal Multiple Access) user groups** under **Imperfect Channel State Information (CSI)**.

---

## 1. Overview & Research Focus

In Next-Generation Wireless Communication (5G/6G), Non-Orthogonal Multiple Access (NOMA) enables multiple users to share the same frequency/time resource block by multiplexing in the power domain. Effective dynamic user grouping is essential for maximizing sum-rate and spectral efficiency. However, in real-world environments, Channel State Information (CSI) is inherently imperfect due to estimation errors, mobility, and feedback delay.

This project implements an **Uncertainty-Aware Semi-Supervised Graph Neural Network** that:
1. Converts tabular wireless CSI datasets into PyTorch Geometric scenario graphs.
2. Explicitly incorporates **CSI uncertainty** as a node feature and reliability filter.
3. Performs **true semi-supervised learning** combining supervised loss on labeled users with dual-perturbation graph consistency loss and confidence + uncertainty-filtered pseudo-labeling on unlabeled users.
4. Predicts NOMA user groups directly using GNN output probabilities.
5. Generates dynamic **NetworkX user graph visualizations**, 2D node embedding plots (PCA & t-SNE), GraphML exports, and CSV predictions.

---

## 2. End-to-End System Pipeline

```text
                 EXTERNAL WIRELESS DATASET (.csv)
                               │
                               ↓
                      DATA PREPROCESSING
            (Feature Scaler fit on train set only)
                               │
                               ↓
                    CSI FEATURE EXTRACTION
           (Channel gain, SNR, distance, CSI Uncertainty)
                               │
                               ↓
                     GRAPH BUILDER (PyG)
          (Nodes = Users, Edges = kNN / Similarity)
                               │
          ┌────────────────────┴────────────────────┐
          │                                         │
          ↓                                         ↓
    LABELED GRAPHS                           UNLABELED GRAPHS
          │                                         │
          ↓                                         ↓
    GNN FORWARD                              GRAPH DUAL PERTURBATIONS
          │                                      (Graph A & B)
          ↓                                         │
  SUPERVISED LOSS                                   ↓
   (Cross Entropy)                           GNN FORWARD (P_A, P_B)
          │                                         │
          │                                         ↓
          │                                 CONSISTENCY LOSS (KL/MSE)
          │                                         +
          │                                 CONFIDENCE & UNCERTAINTY
          │                                 PSEUDO-LABEL LOSS
          │                                         │
          └────────────────────┬────────────────────┘
                               ↓
                    COMPOSITE TOTAL LOSS
            L = L_sup + λ L_cons + μ L_pseudo
                               │
                               ↓
                        GNN TRAINING &
                    MODEL CHECKPOINTING
                               │
                               ↓
                         GNN INFERENCE
                               │
                               ↓
                     GROUP PROBABILITIES
                               │
                               ↓
                    PREDICTED NOMA GROUPS
                               │
                               ↓
                GRAPH & EMBEDDING VISUALIZATION
```

---

## 3. Technology Stack & Dependencies

Strictly built in **Python**:
- **PyTorch & PyTorch Geometric (PyG)**: GNN backbones (`GraphSAGE`, `GCN`, `GAT`), graph convolutions, message passing, and custom loss functions.
- **NetworkX**: Dynamic graph construction, node attribute management, and GraphML export.
- **Matplotlib & Seaborn**: Dynamic NOMA user graph plots, PCA/t-SNE 2D embedding space plots, loss curves, and experiment figures.
- **Scikit-learn**: Feature scaling (without test leakage), PCA, t-SNE, and classification metrics.
- **SciPy**: Hungarian Algorithm (`linear_sum_assignment`) for label permutation alignment.
- **Pandas & NumPy**: Dataset loading, CSI uncertainty computation, and tabular data manipulation.

---

## 4. Project File Structure

```text
NOMA_GNN_Model/
├── config.py                  # Global settings (features, hyperparams, graph methods, paths)
├── requirements.txt           # Required Python packages
├── README.md                  # Comprehensive project documentation
├── train.py                   # Main end-to-end training & experimental execution script
├── predict.py                 # Standalone CLI prediction script for new external datasets
│
├── data/
│   ├── external_dataset.csv   # External wireless dataset (demo dataset auto-generated if absent)
│   └── generate_demo_data.py  # Script for generating realistic wireless CSI benchmark data
│
├── preprocessing/
│   ├── loader.py              # Configurable CSV loader, column validator & stats logger
│   ├── feature_processing.py # Scaler fitting on train split ONLY (no data leakage)
│   └── graph_builder.py       # Conversions of scenario tables into PyG Data graphs
│
├── models/
│   ├── gcn.py                 # PyG GCN architecture
│   ├── graphsage.py           # PyG GraphSAGE architecture (default)
│   ├── gat.py                 # PyG GAT architecture
│   └── uncertainty_gnn.py     # Uncertainty-Aware GNN wrapper & backbone selector
│
├── training/
│   ├── supervised_loss.py     # Masked Cross-Entropy Loss (ignoring -1 unlabeled nodes)
│   ├── consistency_loss.py    # Dual-perturbation graph consistency loss (KL/MSE)
│   ├── pseudo_labeling.py     # Confidence & CSI uncertainty-filtered pseudo-label engine
│   └── trainer.py             # SemiSupervisedTrainer with early checkpointing
│
├── evaluation/
│   ├── metrics.py             # Accuracy, F1, Precision, Recall, Confusion Matrix, Hungarian alignment
│   └── evaluation.py          # Baseline comparisons, Label Efficiency & CSI Uncertainty experiments
│
├── visualization/
│   ├── graph_visualization.py # Dynamic NetworkX + Matplotlib user graph (node size = CSI uncertainty, color = group)
│   ├── embeddings.py          # PCA & t-SNE 2D scatter plots of learned GNN node embeddings
│   └── plots.py               # Training curves, label efficiency curves, CSI impact plots, flowcharts
│
├── tests/
│   └── test_pipeline.py       # Comprehensive unit tests for data loader, PyG GNN, losses, and trainer
│
├── models_saved/              # Saved PyTorch model checkpoints (.pt)
└── results/                   # Evaluation results, predictions, GraphML files, plots
    ├── predictions/           # predicted_groups.csv, prediction_probabilities.csv, node_embeddings.csv, graph_edges.csv
    ├── graphs/                # predicted_graph.graphml, noma_user_group_graph.png
    ├── embeddings/            # gnn_embeddings_pca.png, gnn_embeddings_tsne.png
    └── plots/                 # training_curves.png, label_efficiency.png, csi_uncertainty_impact.png, flowcharts
```

---

## 5. Semi-Supervised Loss Formulations

The total optimization objective for training the GNN is:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{supervised}} + \lambda \cdot \mathcal{L}_{\text{consistency}} + \mu \cdot \mathcal{L}_{\text{pseudo}}$$

### 1. Supervised Loss ($\mathcal{L}_{\text{supervised}}$)
Evaluated only on labeled training nodes ($y_i \neq -1$):

$$\mathcal{L}_{\text{supervised}} = -\frac{1}{|L|} \sum_{i \in L} \sum_{k=1}^{K} y_{i,k} \log P(y_i = k \mid x_i, \mathcal{G})$$

### 2. Dual-Perturbation Consistency Loss ($\mathcal{L}_{\text{consistency}}$)
For unlabeled graphs, two perturbed graph views $\mathcal{G}_A$ and $\mathcal{G}_B$ are generated via random edge dropping and feature masking. The symmetric KL-divergence enforces prediction stability:

$$\mathcal{L}_{\text{consistency}} = \frac{1}{2 \cdot |U|} \sum_{i \in U} \left[ D_{\text{KL}}(P_A(y_i) \parallel P_B(y_i)) + D_{\text{KL}}(P_B(y_i) \parallel P_A(y_i)) \right]$$

### 3. Confidence & CSI Uncertainty Pseudo-Labeling ($\mathcal{L}_{\text{pseudo}}$)
Pseudo-labels $\hat{y}_i = \arg\max P(y_i)$ are assigned for unlabeled sample $i$ if and only if:

$$\max_k P(y_i = k) \ge \tau_{\text{confidence}} \quad \text{AND} \quad \sigma^2_{\text{csi}, i} \le \tau_{\text{uncertainty}}$$

Reliable pseudo-labels contribute to cross-entropy minimization:

$$\mathcal{L}_{\text{pseudo}} = -\frac{1}{|\hat{U}|} \sum_{i \in \hat{U}} \log P(\hat{y}_i \mid x_i, \mathcal{G})$$

---

## 6. How to Run

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```
*(Or use `py -m pip install -r requirements.txt` on Windows)*

### Step 2: Run Unit Tests
Verify pipeline integrity:
```bash
py -m unittest discover -s tests
```

### Step 3: Train Model & Run Research Experiments
```bash
py train.py --epochs 100 --labeled_pct 10
```
- Trains the semi-supervised GNN on 10% labeled training graphs (90% unlabeled).
- Checkpoints best model to `models_saved/best_semi_supervised_gnn.pt`.
- Automatically executes baseline comparisons, label efficiency experiments (5%, 10%, 20%, 30%, 50%), and CSI uncertainty sensitivity experiments.
- Exports all CSV prediction files, GraphML network graphs, 2D embedding plots, and training figures.

### Step 4: Run Standalone Inference Prediction
To predict NOMA user groups on a new external CSV dataset using a trained model:
```bash
py predict.py --input data/external_dataset.csv --model models_saved/best_semi_supervised_gnn.pt --output results/
```

---

## 7. Output Artifacts

After training or inference, output artifacts are saved in `results/`:

1. **Predictions & Metric Summaries (`results/predictions/`)**:
   - `metrics_summary.csv`: Overall test/evaluation metric summary (**Accuracy**, **Precision**, **Recall**, and **F1-Score**).
   - `predicted_groups.csv`: User IDs, predicted NOMA groups, prediction confidence, and CSI uncertainty.
   - `prediction_probabilities.csv`: Softmax probability matrix across all NOMA groups.
   - `node_embeddings.csv`: Learned multi-dimensional GNN node embeddings.
   - `graph_edges.csv`: Constructed user graph edge connections and channel similarity weights.
   - `training_history.csv`: Epoch-by-epoch loss breakdown, validation accuracy, precision, and F1 scores.
   - `label_efficiency.csv`: Grouping accuracy and precision across labeled data ratios (5% to 50%).
   - `baseline_comparison.csv`: Comparative accuracy, precision, and F1 across baseline models.
   - `csi_uncertainty_metrics.csv`: Binned accuracy and precision across CSI uncertainty levels.

2. **Graph Structure & Visualizations (`results/graphs/`)**:
   - `predicted_graph.graphml`: Standard GraphML format readable by Gephi, NetworkX, and Cytoscape.
   - `noma_user_group_graph.png`: NetworkX visualization where node colors represent predicted NOMA groups, node sizes reflect CSI uncertainty levels, and edge thickness indicates channel similarity.

3. **Embeddings & Plots (`results/embeddings/` & `results/plots/`)**:
   - `gnn_embeddings_pca.png`: 2D PCA plot of GNN node embeddings.
   - `gnn_embeddings_tsne.png`: 2D t-SNE plot of GNN node embeddings.
   - `training_curves.png`: Loss curves (Total, Supervised, Consistency, Pseudo-label) and validation accuracy & precision.
   - `label_efficiency.png`: Accuracy & Precision vs Labeled Data Percentage (5% to 50%).
   - `csi_uncertainty_impact.png`: Accuracy, Precision & Confidence vs CSI Uncertainty bins.
   - `model_pipeline_architecture.png` & `semi_supervised_training_architecture.png`: Model flowcharts.

---

## 8. Note on Downstream Transmission Modules

This repository builds **ONLY the machine-learning model** for dynamic NOMA user grouping.

Wireless communication transmission simulators (Successive Interference Cancellation (SIC), Power Allocation, SINR computation, and Sum-Rate data rate calculation) are **downstream modules** outside of this ML core. The GNN user group predictions produced in `predicted_groups.csv` can be fed directly into power allocation and SIC optimization modules in future research stages.
