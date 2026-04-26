import os
import pickle
from src.data.preprocess import preprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEAN_DIR = os.path.join(BASE_DIR, "..", "data", "clean")

METADATA_PATH = os.path.join(CLEAN_DIR, "metadata.csv")
FEATURES_PATH = os.path.join(CLEAN_DIR, "features.csv")
SCALER_PATH = os.path.join(CLEAN_DIR, "scaler.pkl")

def build_features():
    os.makedirs(CLEAN_DIR, exist_ok=True)

    metadata_train, metadata_test, features_train, features_test, scaler, _ = preprocess()

    metadata_train.to_csv(os.path.join(CLEAN_DIR, "metadata_train.csv"), index=False)
    metadata_test.to_csv(os.path.join(CLEAN_DIR, "metadata_test.csv"), index=False)
    features_train.to_csv(os.path.join(CLEAN_DIR, "features_train.csv"), index=False)
    features_test.to_csv(os.path.join(CLEAN_DIR, "features_test.csv"), index=False)

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print(f"Saved {len(metadata_train)} and {len(metadata_test)} tracks to {CLEAN_DIR}")

if __name__ == "__main__":
    build_features()