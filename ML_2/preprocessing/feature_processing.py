import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple


class FeatureScaler:
    """
    Normalizes numerical node features strictly using training-data statistics
    to prevent data leakage into validation/test splits.
    """
    def __init__(self, feature_cols: List[str]):
        self.feature_cols = feature_cols
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, df_train: pd.DataFrame) -> 'FeatureScaler':
        """
        Fits StandardScaler ONLY on the training dataframe split.
        """
        self.scaler.fit(df_train[self.feature_cols].values)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms features in a given dataframe split using fitted training statistics.
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureScaler must be fitted on training data before calling transform().")

        df_transformed = df.copy()
        scaled_vals = self.scaler.transform(df_transformed[self.feature_cols].values)
        df_transformed[self.feature_cols] = scaled_vals
        return df_transformed

    def fit_transform(self, df_train: pd.DataFrame) -> pd.DataFrame:
        """
        Fits on df_train and transforms it.
        """
        self.fit(df_train)
        return self.transform(df_train)
