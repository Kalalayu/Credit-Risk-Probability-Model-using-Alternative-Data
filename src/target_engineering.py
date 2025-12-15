
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import logging

logging.basicConfig(level=logging.INFO)

def calculate_rfm(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Calculating RFM metrics")
    snapshot_date = df["TransactionStartTime"].max() + pd.Timedelta(days=1)
    rfm = (
        df.groupby("CustomerId")
        .agg(
            Recency=("TransactionStartTime", lambda x: (snapshot_date - x.max()).days),
            Frequency=("TransactionId", "count"),
            Monetary=("Amount", "sum"),
        )
        .reset_index()
    )
    return rfm

def cluster_rfm(rfm_df: pd.DataFrame, n_clusters=3, random_state=42) -> pd.DataFrame:
    logging.info("Clustering customers using KMeans")
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[["Recency", "Frequency", "Monetary"]])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    rfm_df["cluster"] = kmeans.fit_predict(rfm_scaled)
    
    return rfm_df

def assign_high_risk(rfm_df: pd.DataFrame, high_risk_cluster: int) -> pd.DataFrame:
    logging.info("Assigning high-risk label")
    rfm_df["is_high_risk"] = (rfm_df["cluster"] == high_risk_cluster).astype(int)
    return rfm_df
