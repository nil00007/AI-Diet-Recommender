import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from .config import *

def load_raw():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"{CSV_PATH} bulunamadı; CSV’yi data/ klasörüne koyun.")
    return pd.read_csv(CSV_PATH)

def build_pipe():
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ("num", StandardScaler(), NUM_COLS)
    ])

def split(df, test=0.2):
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    return train_test_split(X, y, test_size=test,
                            stratify=y, random_state=SEED)
