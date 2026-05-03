import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Work folders config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEAN_DIR = os.path.join(BASE_DIR, "..", "data", "clean")
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "knn_model.pkl")
TF_IDF_PATH = os.path.join(BASE_DIR, "saved_models", "tfidf_vectorizer.pkl")

# liked songs
INPUT_SONGS = [
    "Bohemian Rhapsody",
    "Smells Like Teen Spirit",
    "Back In Black",
    "Enter Sandman",
    "Sweet Child O' Mine",
]

# load of train data
metadata_train = pd.read_csv(os.path.join(CLEAN_DIR, "metadata_train.csv"))
features_train = pd.read_csv(os.path.join(CLEAN_DIR, "features_train.csv"))

if 'Unnamed: 0' in metadata_train.columns:
    metadata_train = metadata_train.drop(columns=['Unnamed: 0'])
if 'Unnamed: 0' in features_train.columns:
    features_train = features_train.drop(columns=['Unnamed: 0'])

# Load of model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(TF_IDF_PATH, "rb") as f:
    tfidf = pickle.load(f)

# Save liked songs
matched = metadata_train[metadata_train['track_name'].isin(INPUT_SONGS)]
matched = matched.drop_duplicates(subset='track_name', keep='first')
matched_indices = matched.index

print(matched[['track_name', 'artists', 'track_genre']].to_string(index=False))

# Take songs of user and create vector
input_features = features_train.loc[matched_indices]
user_profile = pd.DataFrame(
    input_features.mean(axis=0).values.reshape(1, -1),
    columns=features_train.columns
)

# Match songs to knn model
distances, indices = model.kneighbors(user_profile, n_neighbors=15)

#Calculate of top 10 liked songs
recommended = metadata_train.iloc[indices[0]].copy()
recommended['distance'] = distances[0]
recommended = recommended[~recommended['track_name'].isin(INPUT_SONGS)].head(10)

#Result
print("\nRecommendations :")
print(recommended[['track_name', 'artists', 'track_genre', 'distance']].to_string(index=False))

#Charts
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
names = recommended['track_name'].str[:25].tolist()
dists = recommended['distance'].tolist()
bars = axes[0].barh(names, dists, color='skyblue', edgecolor='black')
axes[0].set_xlabel("Distance")
axes[0].set_title("Top 10 ")
axes[0].invert_yaxis()
for bar, val in zip(bars, dists):
    axes[0].text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{val:.2f}', va='center', fontsize=8)

genre_counts = recommended['genre_group'].value_counts()
colors = plt.cm.Set3(np.linspace(0, 1, len(genre_counts)))
axes[1].pie(genre_counts.values, labels=genre_counts.index,
            autopct='%1.1f%%', colors=colors, startangle=90)
axes[1].set_title("Genre chart")

plt.suptitle("KNN recommendations", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()