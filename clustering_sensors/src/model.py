import pickle
from pathlib import Path

import pandas as pd
import numpy as np
import typing

from sklearn.cluster import KMeans
from sklearn.metrics import classification_report


class OutageClusterer:

    sensor_cols = ["Sensor 2", "Sensor 9", "Sensor 13"]
    label_col = "Label"

    def __init__(self, random_state: int = 53) -> None:
        self.random_state = random_state

    def prepare_step1_data(self, data) -> np.array:
        s2 = data["Sensor 2"].values
        s13 = data["Sensor 13"].values
        radius_sq = s2**2 + s13**2
        return np.column_stack([radius_sq])
    
    def _compute_cluster_to_label(self, data_with_cluster: pd.DataFrame, cluster_col: str = "cluster_id", threshold: float = 0.8) -> pd.Series:
        def majority_label(x):
            counts = x.value_counts()
            top_fraction = counts.max() / counts.sum()
            return counts.idxmax() if top_fraction >= threshold else np.nan

        return data_with_cluster.groupby(cluster_col)[self.label_col].agg(majority_label)

    def determine_label(self, data_with_cluster: pd.DataFrame, cluster_col: str = "cluster_id", pred_label_col: str = "pred", threshold: float = 0.8) -> pd.DataFrame:
        cluster_to_label = self._compute_cluster_to_label(data_with_cluster, cluster_col, threshold)
        data_with_cluster[pred_label_col] = data_with_cluster[cluster_col].map(cluster_to_label)
        return data_with_cluster

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "OutageClusterer":
        with open(path, "rb") as f:
            return pickle.load(f)

    def predict(self, data: pd.DataFrame) -> pd.Series:
        result = data.copy()
        result["radius_sq"] = self.prepare_step1_data(result)

        result["cluster_id"] = self.km_step1.predict(
            np.column_stack([result["radius_sq"].values])
        )
        result["pred"] = result["cluster_id"].map(self.step1_cluster_to_label)

        core_mask = result["pred"].isna()
        result.loc[core_mask, "cluster_id"] = self.km_step2.predict(
            result.loc[core_mask, self.sensor_cols].values
        )
        result.loc[core_mask, "pred"] = result.loc[core_mask, "cluster_id"].map(self.step2_cluster_to_label)

        return result["pred"]

    def fit_predict(self, data: pd.DataFrame, report: bool = False) -> pd.DataFrame:
        # Project into higher-dimensional space using squared radius
        result = data.copy()
        result["radius_sq"] = self.prepare_step1_data(result)

        # Separate outer cluster from the core
        self.km_step1 = KMeans(n_clusters=2, random_state=self.random_state, n_init=10)
        result["cluster_id"] = self.km_step1.fit_predict(
            np.column_stack([result["radius_sq"].values])
        )

        # Rename clusters based on majority class, with threshold 80%
        self.step1_cluster_to_label = self._compute_cluster_to_label(result)
        result = self.determine_label(result)

        # Step 2: cluster core with not 80% purity points into the 2 remaining classes
        core_mask = result["pred"].isna()
        X_core = result.loc[core_mask, self.sensor_cols].values
        labels_core = result[self.label_col][core_mask]

        # prepare the labels and centroids for the inner clustering
        core_labels = sorted(labels_core.dropna().unique())
        k_core = len(core_labels)
        seed_centroids_core = np.array(
            [X_core[labels_core.values == l].mean(axis=0) for l in core_labels]
        )

        # fit a second k-means on the remaining instances
        self.km_step2 = KMeans(
            n_clusters=k_core,
            init=seed_centroids_core,
            n_init=1,
            random_state=self.random_state,
        )

        # Predict the clusters of the impure class from step 1
        result.loc[core_mask, "cluster_id"] = self.km_step2.fit_predict(X_core)

        # Now all the clusters should be quite pure in terms of the ground truth label
        self.step2_cluster_to_label = self._compute_cluster_to_label(result.loc[core_mask])
        core_labeled = self.determine_label(result.loc[core_mask].copy())
        result.loc[core_mask, "pred"] = core_labeled["pred"]

        if report:
            print("Check clustering approach for labelled examples")
            labeled_df = result[~result[self.label_col].isna()]
            print(classification_report(
                y_true=labeled_df[self.label_col],
                y_pred=labeled_df["pred"]
            ))

        return result[list(data.columns) + ["pred"]]
        

if __name__ == "__main__": 
    data_path = "data/input/data_sensors.csv"
    data = pd.read_csv(data_path)   

    clusterer = OutageClusterer()
    clustered_data = clusterer.fit_predict(data, report=True)

    clustered_data.to_csv("data/results/predicted_clusters.csv")

