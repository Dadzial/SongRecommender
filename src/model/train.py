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

    tfidf_path = os.path.join(CLEAN_DIR, "tfidf.pkl")
    tfidf = None
    if os.path.exists(tfidf_path):
        with open(tfidf_path, "rb") as f:
            tfidf = pickle.load(f)

    return metadata_train, metadata_test, features_train, features_test, scaler , tfidf

def train_model(features_train,n_neighbors=20):
    model = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean', algorithm='auto')
    model.fit(features_train)
    return model

def save_model(model, tfidf):
    os.makedirs(MODEL_DIR, exist_ok=True)

    path = os.path.join(MODEL_DIR, "knn_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)

    tfidf_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
    with open(tfidf_path, "wb") as f:
        pickle.dump(tfidf, f)

    print(f"Model and Vectorizer saved in {MODEL_DIR}")

if __name__ == "__main__":
    metadata_train, metadata_test, features_train, features_test, scaler, tfidf = load_clean()
    model = train_model(features_train)
    save_model(model, tfidf)
