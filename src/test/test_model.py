import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data.preprocess import preprocess

# Work folders config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "gbr_model.pkl")

# liked songs
INPUT_SONGS = [
    "Jaan 'Nisaar (Arijit)",
    "Tum Hi Ho",
    "Lover",
    "Mann Mera",
    "Ghungroo (From \"War\")",
]

def run_test():
    # Load of model
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Please run train.py first.")
        return

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # Load data
    metadata_train, metadata_test, features_train, features_test, scaler = preprocess(visualize=False)

    # Combine for recommendation pool
    all_metadata = pd.concat([metadata_train, metadata_test])
    all_features = pd.concat([features_train, features_test])

    # Save liked songs
    matched = all_metadata[all_metadata['track_name'].isin(INPUT_SONGS)]
    matched = matched.drop_duplicates(subset='track_name', keep='first')
    print("Songs identified as 'liked':")
    print(matched[['track_name', 'artists', 'track_genre']].to_string(index=False))

    # Predict likedness for all songs (1 = liked, 0 = disliked)
    print("\nCalculating recommendations using Gradient Boosting Regressor...")
    all_metadata['prediction'] = model.predict(all_features)

    # Filter out already liked songs and get top 10 (highest score is better)
    recommended = all_metadata[~all_metadata['track_name'].isin(INPUT_SONGS)].sort_values('prediction', ascending=False).head(10)

    # Result
    print("\nRecommendations (higher score is better):")
    print(recommended[['track_name', 'artists', 'track_genre', 'prediction']].to_string(index=False))

    # Charts
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    names = recommended['track_name'].str[:25].tolist()
    scores = recommended['prediction'].tolist()
    bars = axes[0].barh(names, scores, color='lightgreen', edgecolor='black')
    axes[0].set_xlabel("Predicted Score (closer to 1 is better)")
    axes[0].set_title("Top 10 Recommendations")
    axes[0].invert_yaxis()
    for bar, val in zip(bars, scores):
        axes[0].text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{val:.4f}', va='center', fontsize=8)


    genre_counts = recommended['genre_group'].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(genre_counts)))
    axes[1].pie(genre_counts.values, labels=genre_counts.index,
                autopct='%1.1f%%', colors=colors, startangle=90)
    axes[1].set_title("Genre Distribution of Recommendations")

    plt.suptitle("Gradient Boosting Regressor Recommendations", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_test()
