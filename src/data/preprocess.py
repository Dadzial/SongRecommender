import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.data.fetch import load_raw

METADATA = [
    "track_id",
    "artists",
    "album_name",
    "track_name",
    "popularity",
    "track_genre"
]

FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"
]

def preprocess():
    df = load_raw()
    df = df[METADATA + FEATURES]
    df.dropna(inplace=True)
    df.drop_duplicates(subset="track_id", inplace=True)
    df.reset_index(drop=True, inplace=True)

    metadata = df[METADATA].copy()
    features = df[FEATURES].copy()

    scaler = StandardScaler()
    features_scaled = pd.DataFrame(
        scaler.fit_transform(features),
        columns=FEATURES
    )

    return metadata, features_scaled, scaler

if __name__ == "__main__":
    metadata, features, scaler = preprocess()
    print(metadata.head())
    print(features.head())