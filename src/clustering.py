import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class DriveClusterer:
    """
    Clusters offensive drives into identity types using K-Means.
    
    - Automatically chooses the best number of clusters (k) using silhouette score.
    - Expects drive-level features as created by DrivePreprocessor.
    """

    def __init__(
        self,
        feature_cols=None,
        k_min=2,
        k_max=8,
        random_state=42,
    ):
        """
        Parameters
        ----------
        feature_cols : list of str, optional
            Columns to use as features for clustering.
        k_min : int
            Minimum number of clusters to try.
        k_max : int
            Maximum number of clusters to try.
        random_state : int
            Random seed for K-Means.
        """
        if feature_cols is None:
            feature_cols = [
                "play_count",
                "run_pct",
                "pass_pct",
                "success_rate",
                "avg_epa",
            ]

        self.feature_cols = feature_cols
        self.k_min = k_min
        self.k_max = k_max
        self.random_state = random_state

        self.scaler = None
        self.model = None
        self.best_k = None
        self.k_scores_ = {}  # k -> silhouette score

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Select and scale features from a drives DataFrame.
        """
        # Use only rows with all feature columns present
        X = df[self.feature_cols].copy()

        # Fill any remaining NaNs with 0 for modeling
        X = X.fillna(0.0)

        # Fit scaler if needed
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        return X_scaled

    def fit(self, df: pd.DataFrame):
        """
        Fit K-Means to drive data and automatically choose best k using silhouette score.

        Parameters
        ----------
        df : pd.DataFrame
            Drive-level DataFrame from DrivePreprocessor.
        """
        X_scaled = self._prepare_features(df)

        n_samples = X_scaled.shape[0]
        if n_samples < 3:
            raise ValueError("Not enough drives to cluster (need at least 3).")

        best_score = -1
        best_k = None
        best_model = None

        # Try different k values and compute silhouette scores
        for k in range(self.k_min, self.k_max + 1):
            if n_samples <= k:
                # Can't have more clusters than samples
                continue

            model = KMeans(
                n_clusters=k,
                n_init=10,
                random_state=self.random_state,
            )
            labels = model.fit_predict(X_scaled)

            # Some edge cases can break silhouette_score; catch them
            try:
                score = silhouette_score(X_scaled, labels)
            except Exception:
                continue

            self.k_scores_[k] = score

            if score > best_score:
                best_score = score
                best_k = k
                best_model = model

        if best_model is None:
            raise RuntimeError("Unable to fit a valid K-Means model for any k in range.")

        self.best_k = best_k
        self.model = best_model

        return self

    def assign_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign cluster labels to each drive in a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Drive-level DataFrame.

        Returns
        -------
        pd.DataFrame
            Copy of df with an added 'cluster' column.
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not fitted. Call fit() before assign_clusters().")

        X = df[self.feature_cols].copy().fillna(0.0)
        X_scaled = self.scaler.transform(X)
        labels = self.model.predict(X_scaled)

        out = df.copy()
        out["cluster"] = labels
        return out

    def get_cluster_centers(self) -> pd.DataFrame:
        """
        Get cluster centers in original (un-scaled) feature space.

        Returns
        -------
        pd.DataFrame
            DataFrame with one row per cluster and feature columns.
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        centers_scaled = self.model.cluster_centers_
        centers = self.scaler.inverse_transform(centers_scaled)
        centers_df = pd.DataFrame(centers, columns=self.feature_cols)
        centers_df["cluster"] = range(self.best_k)
        return centers_df

    def get_k_scores(self) -> pd.Series:
        """
        Return silhouette scores for each tested k.

        Returns
        -------
        pd.Series
            Index = k, value = silhouette score.
        """
        return pd.Series(self.k_scores_).sort_index()