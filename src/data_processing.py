# src/data_processing.py

import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO)


def load_data(file_path: str) -> pd.DataFrame:
    logging.info("Loading data from %s", file_path)

    path = Path(file_path)
    if not path.exists():
        logging.error("Data file not found at %s", file_path)
        raise FileNotFoundError(f"Data file not found at {file_path}")

    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Cleaning data")

    df = df.copy()
    df = df.drop_duplicates()

    if 'TransactionStartTime' in df.columns:
        df['TransactionStartTime'] = pd.to_datetime(
            df['TransactionStartTime'], errors='coerce'
        )

    return df


def save_data(df: pd.DataFrame, output_path: str) -> None:
    logging.info("Saving processed data to %s", output_path)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
