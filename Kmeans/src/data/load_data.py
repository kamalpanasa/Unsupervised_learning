import pandas as pd
from src.utils import config

def load_mall_data():
    """
    Loads Mall Customers dataset from local path.
    """
    return pd.read_csv(config.MALL_DATA_PATH)
