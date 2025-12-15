# src/run_data_processing.py

from pathlib import Path
from data_processing import load_data, clean_data, build_features, save_data
from target_engineering import calculate_rfm, cluster_rfm, assign_high_risk
import pandas as pd

print("Starting data processing...")

# Step 1: Load & clean
df = load_data("data/raw/data.csv")
df = clean_data(df)

# Step 2: Build features (Task 3)
agg_df = build_features(df)  

# Step 3: Compute RFM
rfm_df = calculate_rfm(df)

# Step 4: Cluster customers
rfm_df = cluster_rfm(rfm_df, n_clusters=3, random_state=42)

# Step 5: Determine high-risk cluster (inspect cluster summary)
# Example: choose cluster 2 as high-risk (after inspecting cluster means)
high_risk_cluster = 2
rfm_df = assign_high_risk(rfm_df, high_risk_cluster)

# Step 6: Merge target into main features
final_df = agg_df.merge(
    rfm_df[["CustomerId", "is_high_risk"]],
    on="CustomerId",
    how="left"
)

# Step 7: Save processed dataset
output_path = Path("data/processed/model_input_with_target.csv")
save_data(final_df, output_path)

print(f"Task 4 completed. Data saved to {output_path}")
