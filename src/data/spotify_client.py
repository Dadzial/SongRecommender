import spotipy
from spotipy.oauth2 import SpotifyOAuth

from src.config.spotify_config import (
    SPOTIFY_CLIENT_ID,
    SPOTIFY_CLIENT_SECRET
)

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET,
    redirect_uri="http://127.0.0.1:8888/callback",
    scope="user-read-private"
))