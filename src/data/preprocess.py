import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from src.data.fetch import load_raw
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import os

#METADATA FOR SONGS
METADATA = [
    "track_id",
    "artists",
    "album_name",
    "track_name",
    "popularity",
    "track_genre"
]

#FEATURES FOR SONGS
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

GENRE_MAPPING = {
    #POP
    'pop': 'pop',
    'dance': 'pop',
    'power-pop': 'pop',
    'synth-pop': 'pop',
    'indie-pop': 'pop',
    'j-pop': 'pop',
    'k-pop': 'pop',
    'mandopop': 'pop',
    'cantopop': 'pop',
    'pop-film': 'pop',
    'party': 'pop',
    'happy': 'pop',

    # HIP-HOP / R&B
    'hip-hop': 'hip-hop',
    'r-n-b': 'hip-hop',
    'dancehall': 'hip-hop',
    'trip-hop': 'hip-hop',

    # ROCK
    'rock': 'rock',
    'alt-rock': 'rock',
    'alternative': 'rock',
    'hard-rock': 'rock',
    'punk-rock': 'rock',
    'punk': 'rock',
    'emo': 'rock',
    'grunge': 'rock',
    'indie': 'rock',
    'garage': 'rock',
    'rockabilly': 'rock',
    'rock-n-roll': 'rock',
    'psych-rock': 'rock',
    'british': 'rock',
    'guitar': 'rock',

    #ROCK
    'metal': 'metal',
    'heavy-metal': 'metal',
    'black-metal': 'metal',
    'death-metal': 'metal',
    'metalcore': 'metal',
    'hardcore': 'metal',
    'grindcore': 'metal',
    'goth': 'metal',
    'industrial': 'metal',

    # METAL
    'edm': 'electronic',
    'electronic': 'electronic',
    'electro': 'electronic',
    'house': 'electronic',
    'deep-house': 'electronic',
    'techno': 'electronic',
    'trance': 'electronic',
    'dubstep': 'electronic',
    'drum-and-bass': 'electronic',
    'breakbeat': 'electronic',
    'chicago-house': 'electronic',
    'detroit-techno': 'electronic',
    'minimal-techno': 'electronic',
    'club': 'electronic',
    'disco': 'electronic',
    'hardstyle': 'electronic',
    'progressive-house': 'electronic',
    'idm': 'electronic',

    # ELECTRONIC / DANCE
    'acoustic': 'acoustic',
    'folk': 'acoustic',
    'singer-songwriter': 'acoustic',
    'songwriter': 'acoustic',
    'country': 'acoustic',
    'bluegrass': 'acoustic',
    'honky-tonk': 'acoustic',
    'sad': 'acoustic',

    # ACOUSTIC / FOLK / COUNTRY
    'jazz': 'jazz-soul',
    'blues': 'jazz-soul',
    'soul': 'jazz-soul',
    'funk': 'jazz-soul',
    'gospel': 'jazz-soul',
    'groove': 'jazz-soul',

    # JAZZ / BLUES / SOUL / FUNK
    'classical': 'classical',
    'opera': 'classical',
    'piano': 'classical',
    'ambient': 'classical',
    'sleep': 'classical',
    'study': 'classical',
    'new-age': 'classical',
    'chill': 'classical',

    # LATIN
    'latin': 'latin',
    'latino': 'latin',
    'reggaeton': 'latin',
    'salsa': 'latin',
    'samba': 'latin',
    'bossanova': 'latin',
    'sertanejo': 'latin',
    'pagode': 'latin',
    'mpb': 'latin',
    'forro': 'latin',
    'tango': 'latin',
    'spanish': 'latin',
    'brazil': 'latin',
    'romance': 'latin',

    # REGGAE / SKA
    'reggae': 'reggae',
    'ska': 'reggae',
    'dub': 'reggae',

    # WORLD
    'world-music': 'world',
    'turkish': 'world',
    'afrobeat': 'world',
    'iranian': 'world',
    'indian': 'world',
    'j-rock': 'world',
    'j-dance': 'world',
    'j-idol': 'world',
    'malay': 'world',
    'french': 'world',
    'german': 'world',
    'swedish': 'world',

    # KIDS / COMEDY
    'children': 'inne',
    'kids': 'inne',
    'disney': 'inne',
    'comedy': 'inne',
    'anime': 'inne',
    'show-tunes': 'inne',
}


def plot_distributions(features_before, features_after, features_to_transform):
    fig, axes = plt.subplots(len(features_to_transform), 2, figsize=(12, len(features_to_transform) * 3))

    for i, col in enumerate(features_to_transform):

        axes[i, 0].hist(features_before[col], bins=30, color='skyblue', edgecolor='black')
        axes[i, 0].set_title(f'{col} — Before')
        axes[i, 0].grid(False)


        axes[i, 1].hist(features_after[col], bins=30, color='lightgreen', edgecolor='black')
        axes[i, 1].set_title(f'{col} — After log1p')
        axes[i, 1].grid(False)

    plt.suptitle("Distributions after and before transform", fontsize=14)
    plt.tight_layout()
    plt.show()


def preprocess(visualize=True, random_state=42):
    df = load_raw()
    df = df[METADATA + FEATURES]
    df.dropna(inplace=True)
    df.drop_duplicates(subset="track_id", inplace=True)
    df.reset_index(drop=True, inplace=True)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CLEAN_PATH = os.path.join(BASE_DIR, "clean", "tfidf.pkl")

    df['genre_group'] = df['track_genre'].map(GENRE_MAPPING).fillna('other')

    features = df[FEATURES].copy()
    skew_limit = 0.75
    skew_values = features.skew()
    features_to_transform = skew_values[abs(skew_values) > skew_limit].index.tolist()
    for col in features_to_transform:
        if features[col].min() >= 0:
            features[col] = np.log1p(features[col])

    df_train, df_test, X_train_audio, X_test_audio = train_test_split(
        df, features, test_size=0.2, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_audio_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_audio),
        columns=FEATURES,
        index=df_train.index
    )
    X_test_audio_scaled = pd.DataFrame(
        scaler.transform(X_test_audio),
        columns=FEATURES,
        index=df_test.index
    )

    tfidf = TfidfVectorizer()
    genre_train_matrix = tfidf.fit_transform(df_train['genre_group'])
    genre_test_matrix = tfidf.transform(df_test['genre_group'])

    genre_train_df = pd.DataFrame(
        genre_train_matrix.toarray(),
        columns=tfidf.get_feature_names_out(),
        index=df_train.index
    )
    genre_test_df = pd.DataFrame(
        genre_test_matrix.toarray(),
        columns=tfidf.get_feature_names_out(),
        index=df_test.index
    )

    features_train_final = pd.concat([X_train_audio_scaled, genre_train_df], axis=1)
    features_test_final = pd.concat([X_test_audio_scaled, genre_test_df], axis=1)


    metadata_train = df_train[METADATA + ['genre_group']]
    metadata_test = df_test[METADATA + ['genre_group']]

    with open(CLEAN_PATH, "wb") as f:
        pickle.dump(tfidf, f)

    return metadata_train, metadata_test, features_train_final, features_test_final, scaler , tfidf


def plot_correlations(features_scaled):
    plt.figure(figsize=(12, 6))
    df_correl = features_scaled if isinstance(features_scaled, pd.DataFrame) else pd.DataFrame(features_scaled)
    sns.heatmap(df_correl.corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title("Correlation Matrix")
    plt.show()

if __name__ == "__main__":
    metadata_train, metadata_test, features_train, features_test, scaler ,tfidf = preprocess()

    print(metadata_train.head())
    print(features_train.head())
    plot_correlations(features_train)