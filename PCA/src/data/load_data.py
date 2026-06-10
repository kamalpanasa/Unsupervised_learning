import pandas as pd
from src.utils import config

def load_heroes_data():
    """
    Loads raw Superhero dataset from local data folder.
    """
    return pd.read_csv(config.HEROES_DATA_PATH)
