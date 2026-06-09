import pandas as pd
from src.utils import config

def clean_spotify_data(df):
    """
    Cleans Spotify features.
    """
    return df.dropna(subset=config.AUDIO_FEATURES).copy()
