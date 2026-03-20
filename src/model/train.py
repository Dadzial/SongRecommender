import os
import pickle
import pandas as pd
from sklearn.neighbors import NearestNeighbors

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEAN_DIR = os.path.join(BASE_DIR, "..", "data", "clean")
MODEL_DIR = os.path.join(BASE_DIR, "..", "test", "saved_models")

def load_clean():
    metadata_train = pd.read_csv(os.path.join(CLEAN_DIR, "metadata_train.csv"))
    metadata_test  = pd.read_csv(os.path.join(CLEAN_DIR, "metadata_test.csv"))
    features_train = pd.read_csv(os.path.join(CLEAN_DIR, "features_train.csv"))
    features_test  = pd.read_csv(os.path.join(CLEAN_DIR, "features_test.csv"))

    with open(os.path.join(CLEAN_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    return metadata_train, metadata_test, features_train, features_test, scaler

def train_model(features_train,n_neighbors=20):
    model = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean', algorithm='auto')
    model.fit(features_train)
    return model

def save_model(model):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, "knn_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model is saved {path}")

if __name__ == "__main__":
    metadata_train, metadata_test, features_train, features_test, scaler = load_clean()
    model = train_model(features_train)
    save_model(model)
