import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

from src.config.spotify_config import (
    SPOTIFY_CLIENT_ID,
    SPOTIFY_CLIENT_SECRET
)

auth_manager = SpotifyClientCredentials(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET
)

sp = spotipy.Spotify(auth_manager=auth_manager)