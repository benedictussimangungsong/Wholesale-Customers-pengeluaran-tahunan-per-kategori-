import os
import io
import requests
import pandas as pd

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv"
LOCAL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "Wholesale customers data.csv")

FEATURE_COLS = [
    "Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicassen"
]

def ensure_dataset(local_path: str = LOCAL_PATH, url: str = UCI_URL) -> str:
    """
    Ensure the Wholesale Customers dataset exists locally.
    If missing, download it from UCI and save to `data/`.
    Returns the local file path.
    """
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if not os.path.exists(local_path):
        print("Dataset not found locally. Downloading from UCI...")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        # The UCI file uses ';' or ',' depending on mirror; we store exactly as bytes
        with open(local_path, "wb") as f:
            f.write(resp.content)
        print(f"Saved dataset to {local_path}")
    else:
        print(f"Dataset already exists at {local_path}")
    return local_path

def load_data(local_path: str = LOCAL_PATH) -> pd.DataFrame:
    """
    Load dataset into a DataFrame with robust delimiter detection.
    """
    # Try comma first, then semicolon
    try:
        df = pd.read_csv(local_path)
    except Exception:
        df = pd.read_csv(local_path, sep=";")
    # Standardize column names if needed
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    # Handle historical naming
    rename_map = {"Detergents_Paper":"Detergents_Paper", "Delicassen":"Delicassen"}
    df.rename(columns=rename_map, inplace=True)
    # Sometimes 'Channel' and 'Region' exist in some versions — keep if present
    return df

def describe_numeric(df: pd.DataFrame) -> dict:
    desc = df.describe(include="all").to_dict()
    return desc