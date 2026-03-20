import pandas as pd
import os

RAW_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw", "dataset.csv")

def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)
    return df

if __name__ == "__main__":
    df = load_raw()
    print(f"Loaded {len(df)} rows")
    print(df.dtypes)
    print(df.head())