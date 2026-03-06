from src.data.fetch import search_tracks

tracks = search_tracks("Daft Punk")

for track in tracks:
    print(track["name"])