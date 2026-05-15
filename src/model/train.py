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

# load data with charts
def load_data():
    return preprocess(visualize=False)

# training model
def train_model(X_train, y_train):
    # use model gbr
    model = GradientBoostingRegressor(
        n_estimators=200, 
        learning_rate=0.05, 
        max_depth=5, 
        min_samples_split=5,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

# add metrics for evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"\nModel Evaluation Metrics (Similarity Prediction):")
    print(f"R2 Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    return r2, mae, rmse

def residual_and_fit_plot(model, X_test, y_test):
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    plt.figure(figsize=(12, 5))

    # Residual plot
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted Similarity')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')

    # Fit plot
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Similarity')
    plt.ylabel('Predicted Similarity')
    plt.title('Actual vs Predicted Similarity')

    plt.tight_layout()
    plt.show()

# save model to file
def save_model(model):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, "gbr_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel is saved {path}")

# train model
def run_training(liked_songs=None):
    metadata_train, metadata_test, features_train, features_test, scaler = load_data()

    if liked_songs is None:
        liked_genres = ['rock', 'metal']
        y_train = metadata_train['genre_group'].apply(lambda x: 1.0 if x in liked_genres else 0.0)
        y_test = metadata_test['genre_group'].apply(lambda x: 1.0 if x in liked_genres else 0.0)
    else:
        all_metadata = pd.concat([metadata_train, metadata_test])
        all_features = pd.concat([features_train, features_test])
        
        user_songs_idx = all_metadata[all_metadata['track_name'].isin(liked_songs)].index
        if len(user_songs_idx) == 0:
            print("No liked songs found in database. Using default genre logic.")
            liked_genres = ['rock', 'metal']
            y_train = metadata_train['genre_group'].apply(lambda x: 1.0 if x in liked_genres else 0.0)
            y_test = metadata_test['genre_group'].apply(lambda x: 1.0 if x in liked_genres else 0.0)
        else:
            user_profile = all_features.loc[user_songs_idx].mean(axis=0)

            # calculate similarity between user profile and all songs
            def calculate_similarity(features_df, profile):
                feat_values = features_df.values.astype(np.float64)
                prof_values = profile.values.astype(np.float64)

                dist = np.linalg.norm(feat_values - prof_values, axis=1)
                sim = 1 / (1 + dist)
                return sim

            y_train = calculate_similarity(features_train, user_profile)
            y_test = calculate_similarity(features_test, user_profile)
            print(f"Training based on similarity to {len(user_songs_idx)} liked songs.")

    print("Training Gradient Boosting Regressor...")
    model = train_model(features_train, y_train)
    
    evaluate_model(model, features_test, y_test)
    residual_and_fit_plot(model, features_test, y_test)
    save_model(model)

if __name__ == "__main__":
    try:
        from src.test.test_model import INPUT_SONGS
        run_training(INPUT_SONGS)
    except ImportError:
        run_training()
