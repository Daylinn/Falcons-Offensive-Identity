import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class DrivePCA:
    """
    Performs PCA on drive-level features and returns 2D coordinates
    for visualization of cluster separation.
    """

    def __init__(self, feature_cols=None):
        if feature_cols is None:
            feature_cols = [
                "play_count",
                "run_pct",
                "pass_pct",
                "success_rate",
                "avg_epa",
            ]
        self.feature_cols = feature_cols
        self.scaler = None
        self.pca = None

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run PCA and return a dataframe with PCA components.
        """
        X = df[self.feature_cols].copy().fillna(0.0)

        # scale
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # pca
        self.pca = PCA(n_components=2)
        coords = self.pca.fit_transform(X_scaled)

        out = df.copy()
        out["PC1"] = coords[:, 0]
        out["PC2"] = coords[:, 1]
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply previously fitted PCA to a new dataframe.
        """
        if self.scaler is None or self.pca is None:
            raise RuntimeError("Call fit_transform on a baseline dataset first.")

        X = df[self.feature_cols].copy().fillna(0.0)
        X_scaled = self.scaler.transform(X)
        coords = self.pca.transform(X_scaled)

        out = df.copy()
        out["PC1"] = coords[:, 0]
        out["PC2"] = coords[:, 1]
        return out