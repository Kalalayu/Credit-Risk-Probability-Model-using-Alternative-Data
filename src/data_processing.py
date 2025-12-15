# Step 1 — Load & clean raw data
import pandas as pd
from pathlib import Path

def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"], errors="coerce"
    )
    return df
def save_data(df: pd.DataFrame, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

# Step 2 - Extract datetime features
def extract_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TransactionHour"] = df["TransactionStartTime"].dt.hour
    df["TransactionDay"] = df["TransactionStartTime"].dt.day
    df["TransactionMonth"] = df["TransactionStartTime"].dt.month
    df["TransactionYear"] = df["TransactionStartTime"].dt.year
    return df

# Step 3 — Aggregate customer-level features
def aggregate_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    agg_df = df.groupby("CustomerId").agg(
        TotalTransactionAmount=("Amount", "sum"),
        AverageTransactionAmount=("Amount", "mean"),
        TransactionCount=("TransactionId", "count"),
        StdTransactionAmount=("Amount", "std"),
        ChannelId=("ChannelId", "first"),
        ProductCategory=("ProductCategory", "first"),
        ProviderId=("ProviderId", "first"),
        CurrencyCode=("CurrencyCode", "first"),
    ).reset_index()

    return agg_df
# Step 4 — Create proxy default variable (required for WoE)
def create_proxy_default(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["proxy_default"] = (df["TransactionCount"] < 3).astype(int)
    return df

# Step 5 — Encode categorical variables
from sklearn.preprocessing import LabelEncoder

def encode_categoricals(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    df = df.copy()
    for col in cat_cols:
        df[col] = df[col].fillna("Missing")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df
# Step 6 — Build sklearn Pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

NUMERIC_FEATURES = [
    "TotalTransactionAmount",
    "AverageTransactionAmount",
    "TransactionCount",
    "StdTransactionAmount",
]

def build_preprocessing_pipeline():
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES)
        ]
    )

    return preprocessor

# Step 7 — Orchestrate feature building
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = extract_datetime_features(df)
    agg_df = aggregate_customer_features(df)
    agg_df = create_proxy_default(agg_df)
    return agg_df

