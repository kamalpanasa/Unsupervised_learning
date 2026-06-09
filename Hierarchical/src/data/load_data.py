import os
import pandas as pd
from src.utils import config

def download_pokemon_data():
    if not os.path.exists(config.POKEMON_PATH):
        print("Downloading Pokemon dataset...")
        url = "https://raw.githubusercontent.com/KeithGalli/pandas/master/pokemon_data.csv"
        df = pd.read_csv(url)
        df.to_csv(config.POKEMON_PATH, index=False)
        print("Dataset downloaded and saved.")
    else:
        print("Pokemon dataset already exists.")

def load_raw_data():
    download_pokemon_data()
    return pd.read_csv(config.POKEMON_PATH)
