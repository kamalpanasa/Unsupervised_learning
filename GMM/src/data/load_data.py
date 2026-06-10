import pandas as pd
from src.utils import config

def load_spotify_data():
    """
    Loads raw Spotify dataset from local data folder.
    """
    return pd.read_csv(config.SPOTIFY_DATA_PATH)
