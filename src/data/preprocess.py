import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.data.fetch import load_raw
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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


def preprocess(visualize=True,random_state=42):
    df = load_raw()
    df = df[METADATA + FEATURES]
    df.dropna(inplace=True)
    df.drop_duplicates(subset="track_id", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['genre_group'] = df['track_genre'].map(GENRE_MAPPING).fillna('inne')


    metadata = df[METADATA + ['genre_group']].copy()
    features = df[FEATURES].copy()

    skew_limit = 0.75
    skew_values = features.skew()
    features_to_transform = skew_values[abs(skew_values) > skew_limit].index.tolist()

    features_to_transform = [
        col for col in features_to_transform
        if features[col].min() >= 0
    ]

    print(f"Features type to transform (skew > {skew_limit}):")
    print(skew_values[abs(skew_values) > skew_limit])

    features_before = features.copy()

    for col in features_to_transform:
        features[col] = np.log1p(features[col])

    if visualize and features_to_transform:
        plot_distributions(features_before, features, features_to_transform)

    features_train, features_test, metadata_train, metadata_test = train_test_split(
        features, metadata, test_size=0.2, random_state=random_state
    )

    scaler = StandardScaler()
    features_train_scaled = pd.DataFrame(
        scaler.fit_transform(features_train),
        columns=features.columns,
        index=features_train.index
    )
    features_test_scaled = pd.DataFrame(
        scaler.transform(features_test),
        columns=features.columns,
        index=features_test.index
    )

    return metadata_train, metadata_test, features_train_scaled, features_test_scaled, scaler


def plot_correlations(features_scaled):
    plt.figure(figsize=(12, 6))
    df_correl = features_scaled if isinstance(features_scaled, pd.DataFrame) else pd.DataFrame(features_scaled)
    sns.heatmap(df_correl.corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title("Correlation Matrix")
    plt.show()

if __name__ == "__main__":
    metadata_train, metadata_test, features_train, features_test, scaler = preprocess()
    print(metadata_train.head())
    print(features_train.head())
    plot_correlations(features_train)