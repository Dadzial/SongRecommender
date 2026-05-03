import os
import pickle
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from src.data.preprocess import preprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "test", "saved_models")

def load_data():
    return preprocess(visualize=False)

def train_model(X_train, y_train):
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"\nModel Evaluation Metrics:")
    print(f"R2 Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Residual Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot (Errors)')
    plt.show()
    
    return r2, mae, rmse

def save_model(model):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, "gbr_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel is saved {path}")

def run_training(liked_songs=None):
    metadata_train, metadata_test, features_train, features_test, scaler = load_data()

    if liked_songs is None:
        liked_genres = ['rock', 'metal']
    else:
        all_metadata = pd.concat([metadata_train, metadata_test])
        user_songs = all_metadata[all_metadata['track_name'].isin(liked_songs)]
        liked_genres = user_songs['genre_group'].unique().tolist()
        print(f"Detected liked genre groups: {liked_genres}")


    y_train = metadata_train['genre_group'].apply(lambda x: 1 if x in liked_genres else 0)
    y_test = metadata_test['genre_group'].apply(lambda x: 1 if x in liked_genres else 0)

    print("Training Gradient Boosting Regressor...")
    model = train_model(features_train, y_train)
    
    evaluate_model(model, features_test, y_test)
    save_model(model)

if __name__ == "__main__":
    try:
        from src.test.test_model import INPUT_SONGS
        run_training(INPUT_SONGS)
    except ImportError:
        run_training()
