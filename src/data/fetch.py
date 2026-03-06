from src.data.spotify_client import sp

def search_tracks(query, limit=5):
    results = sp.search(q=query, type="track", limit=limit)
    return results["tracks"]["items"]