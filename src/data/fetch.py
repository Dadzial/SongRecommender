import pandas as pd

COLUMNS = [
    "track_id",
    "artists",
    "album_name",
    "track_name",
    "danceability",
    "energy",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
    "track_genre"
]


def read_csv_to_df():
    data_df = pd.read_csv('raw/dataset.csv')
    data_df = data_df[COLUMNS]
    genre_dummies = pd.get_dummies(data_df["track_genre"], prefix="genre")
    data_df = pd.concat([data_df, genre_dummies], axis=1)

    return data_df


if __name__ == "__main__":
    df = read_csv_to_df()