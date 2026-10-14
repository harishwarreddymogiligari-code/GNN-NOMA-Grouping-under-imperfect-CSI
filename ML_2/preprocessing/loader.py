import os
import pandas as pd
import numpy as np
import config


def load_raw_dataset(csv_path: str = None) -> pd.DataFrame:
    """
    Loads and validates external CSV dataset for NOMA User Grouping.
    
    Checks configurable feature columns. If 'csi_uncertainty' is absent,
    attempts to derive it automatically from true and estimated CSI columns.
    Raises clear ValueError if essential columns are missing.
    """
    if csv_path is None:
        csv_path = config.DATASET_PATH

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset file not found at: {csv_path}. "
            f"Please verify path in config.py or provide a valid CSV path."
        )

    df = pd.read_csv(csv_path)

    # Check for CSI uncertainty column, compute if missing
    if "csi_uncertainty" not in df.columns:
        print("[DATA LOADER] 'csi_uncertainty' column not found. Attempting derivation from CSI features...")
        
        # Method 1: Derive from complex components (real & imag)
        has_real_imag = all(col in df.columns for col in [
            config.TRUE_CSI_REAL_COL, config.TRUE_CSI_IMAG_COL,
            config.EST_CSI_REAL_COL, config.EST_CSI_IMAG_COL
        ])
        
        # Method 2: Derive from scalar channel gain
        has_gains = all(col in df.columns for col in [
            config.TRUE_CSI_GAIN_COL, config.EST_CSI_GAIN_COL
        ])

        if has_real_imag:
            err_r = df[config.TRUE_CSI_REAL_COL] - df[config.EST_CSI_REAL_COL]
            err_i = df[config.TRUE_CSI_IMAG_COL] - df[config.EST_CSI_IMAG_COL]
            df["csi_error"] = np.sqrt(err_r**2 + err_i**2)
            df["csi_uncertainty"] = err_r**2 + err_i**2
            print("[DATA LOADER] Computed 'csi_uncertainty' from real/imag CSI components.")
        elif has_gains:
            err = df[config.TRUE_CSI_GAIN_COL] - df[config.EST_CSI_GAIN_COL]
            df["csi_error"] = np.abs(err)
            df["csi_uncertainty"] = err**2
            print("[DATA LOADER] Computed 'csi_uncertainty' from channel gain differences.")
        else:
            raise ValueError(
                "Missing required column 'csi_uncertainty' and unable to derive it. "
                f"Dataset must contain either 'csi_uncertainty' OR true/estimated CSI columns "
                f"({config.TRUE_CSI_REAL_COL}, {config.EST_CSI_REAL_COL}) or "
                f"({config.TRUE_CSI_GAIN_COL}, {config.EST_CSI_GAIN_COL})."
            )

    # Validate all requested feature columns
    missing_features = [col for col in config.FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        raise ValueError(
            f"The dataset at '{csv_path}' is missing configured feature columns: {missing_features}. "
            f"Available columns: {list(df.columns)}. Update config.FEATURE_COLUMNS or fix dataset schema."
        )

    # Validate label column (optional if purely unlabeled inference data)
    if config.LABEL_COLUMN not in df.columns:
        print(f"[DATA LOADER WARNING] Label column '{config.LABEL_COLUMN}' absent. Initializing unlabeled (-1).")
        df[config.LABEL_COLUMN] = -1

    return df


def print_dataset_info(df: pd.DataFrame, num_labeled: int = None, num_unlabeled: int = None):
    """
    Prints diagnostic dataset summary formatted to match project specifications.
    """
    total_samples = len(df)
    features_count = len(config.FEATURE_COLUMNS)
    
    if num_labeled is None or num_unlabeled is None:
        if config.LABEL_COLUMN in df.columns:
            num_labeled = int((df[config.LABEL_COLUMN] != -1).sum())
            num_unlabeled = total_samples - num_labeled
        else:
            num_labeled = 0
            num_unlabeled = total_samples

    n_scenarios = df[config.SCENARIO_COLUMN].nunique() if config.SCENARIO_COLUMN in df.columns else 1
    num_groups = config.NUM_GROUPS
    missing_vals = df[config.FEATURE_COLUMNS].isna().sum().sum()

    print("========================================")
    print("DATASET INFORMATION")
    print("========================================")
    print(f"Samples: {total_samples:,}")
    print(f"Scenarios / Graphs: {n_scenarios:,}")
    print(f"Features: {features_count}")
    print(f"Labeled: {num_labeled:,}")
    print(f"Unlabeled: {num_unlabeled:,}")
    print(f"NOMA groups: {num_groups}")
    print(f"Missing values: {missing_vals}")
    print("========================================")
