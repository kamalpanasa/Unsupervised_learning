import pandas as pd
from src.utils import config

def load_uber_data():
    """
    Loads raw Uber dataset from local data folder.
    """
    return pd.read_csv(config.UBER_DATA_PATH)
