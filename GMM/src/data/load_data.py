import os
import pandas as pd
from src.utils import config

def download_spotify_data():
    if not os.path.exists(config.SPOTIFY_DATA_PATH):
        print("Downloading Spotify dataset...")
        url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv"
        df_raw = pd.read_csv(url)
        
        # Select relevant columns and clean
        cols_to_keep = [
            "track_id", "track_name", "track_artist", "playlist_genre",
            "danceability", "energy", "key", "loudness", "mode", "speechiness",
            "acousticness", "instrumentalness", "liveness", "valence", "tempo"
        ]
        df_cleaned = df_raw[cols_to_keep].dropna().drop_duplicates(subset=["track_name", "track_artist"]).reset_index(drop=True)
        # Sample 5,000 songs to keep it fast
        df_sampled = df_cleaned.sample(n=5000, random_state=42).reset_index(drop=True)
        df_sampled.to_csv(config.SPOTIFY_DATA_PATH, index=False)
        print("Spotify dataset downloaded and sampled.")
    else:
        print("Spotify dataset already exists.")

def load_spotify_data():
    download_spotify_data()
    return pd.read_csv(config.SPOTIFY_DATA_PATH)
