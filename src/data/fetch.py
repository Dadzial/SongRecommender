import pandas as pd
import os

# path to the raw dataset
RAW_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw", "dataset.csv")

# load dataset
def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)
    return df

# display dataset
if __name__ == "__main__":
    df = load_raw()
    print(f"Loaded {len(df)} rows")
    print(df.dtypes)
    print(df.head())
    print(df['track_genre'].unique().tolist())
    print(df['track_genre'].value_counts())