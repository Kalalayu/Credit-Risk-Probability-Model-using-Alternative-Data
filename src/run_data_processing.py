from pathlib import Path
import pandas as pd
from data_processing import (
    load_data,
    clean_data,
    build_features,
    encode_categoricals,
    build_preprocessing_pipeline,
    save_data
)

print("Starting feature engineering...")

df = load_data("data/raw/data.csv")
df = clean_data(df)

agg_df = build_features(df)

cat_cols = ["ChannelId", "ProductCategory", "ProviderId", "CurrencyCode"]
cat_cols = [c for c in cat_cols if c in agg_df.columns]

agg_df = encode_categoricals(agg_df, cat_cols)

pipeline = build_preprocessing_pipeline()
X_numeric = pipeline.fit_transform(agg_df)

processed_df = pd.DataFrame(
    X_numeric,
    columns=pipeline.get_feature_names_out()
)

output_path = Path("data/processed/model_input.csv")
save_data(processed_df, output_path)

print("Feature engineering completed successfully.")
