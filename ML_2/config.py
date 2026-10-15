import os

# ==========================================
# DATASET & FEATURE CONFIGURATION
# ==========================================
DATASET_PATH = os.path.join("data", "external_dataset.csv")

# Feature columns to extract for node features
FEATURE_COLUMNS = [
    "channel_gain",
    "distance",
    "csi_uncertainty",
    "snr",
    "channel_real",
    "channel_imag"
]

LABEL_COLUMN = "user_group"
SCENARIO_COLUMN = "scenario_id"
USER_ID_COLUMN = "user_id"

# Columns used for automatic CSI uncertainty computation if missing
TRUE_CSI_REAL_COL = "channel_real"
TRUE_CSI_IMAG_COL = "channel_imag"
EST_CSI_REAL_COL = "estimated_channel_real"
EST_CSI_IMAG_COL = "estimated_channel_imag"
TRUE_CSI_GAIN_COL = "channel_gain"
EST_CSI_GAIN_COL = "estimated_channel"

# Number of NOMA user groups / clusters
NUM_GROUPS = 4

# ==========================================
# GRAPH CONSTRUCTION CONFIGURATION
# ==========================================
# Supported methods: "knn", "channel_similarity", "fully_connected"
GRAPH_METHOD = "knn"
K_NEIGHBORS = 4

# ==========================================
# GNN MODEL ARCHITECTURE CONFIGURATION
# ==========================================
# Supported backbones: "GraphSAGE", "GCN", "GAT"
GNN_TYPE = "GraphSAGE"

HIDDEN_DIM = 64
NUM_LAYERS = 3
DROPOUT = 0.2

# ==========================================
# SEMI-SUPERVISED & UNCERTAINTY HYPERPARAMETERS
# ==========================================
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
EPOCHS = 100
BATCH_SIZE = 16

# Weights for semi-supervised loss components
LAMBDA_CONSISTENCY = 1.0   # Weight for graph dual-perturbation consistency loss
MU_PSEUDO = 0.5            # Weight for confidence-uncertainty pseudo-label loss

# Confidence and uncertainty thresholds for pseudo-labeling
CONFIDENCE_THRESHOLD = 0.90
UNCERTAINTY_THRESHOLD = 0.50  # Normalized percentile or raw uncertainty cut-off

# Graph perturbation rates for consistency loss
PERTURB_EDGE_DROP_RATE = 0.15
PERTURB_FEATURE_MASK_RATE = 0.10

# Default training label split ratio for semi-supervised experiments
LABELED_PERCENTAGE = 10  # e.g., 10% labeled training samples (Case B/C)

# Data Split Ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ==========================================
# DIRECTORIES & REPRODUCIBILITY
# ==========================================
MODEL_SAVE_DIR = "models_saved"
RESULTS_DIR = "results"
RANDOM_SEED = 42
