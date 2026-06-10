import pandas as pd
from src.utils import config

def load_raw_data():
    """
    Loads raw Pokemon dataset from local data folder.
    """
    return pd.read_csv(config.POKEMON_PATH)
