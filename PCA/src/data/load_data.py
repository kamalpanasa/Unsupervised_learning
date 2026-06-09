import os
import pandas as pd
from src.utils import config

def download_heroes_data():
    if not os.path.exists(config.HEROES_DATA_PATH):
        print("Downloading Superhero power stats dataset...")
        # Note the verified url from akumar-99
        url = "https://raw.githubusercontent.com/akumar-99/Programming-II-Python/master/characters_stats.csv"
        df_raw = pd.read_csv(url)
        df_raw.to_csv(config.HEROES_DATA_PATH, index=False)
        print("Superhero dataset downloaded and saved.")
    else:
        print("Superhero dataset already exists.")

def load_heroes_data():
    download_heroes_data()
    return pd.read_csv(config.HEROES_DATA_PATH)
