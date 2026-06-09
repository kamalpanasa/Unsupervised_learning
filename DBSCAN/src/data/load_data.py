import os
import pandas as pd
from src.utils import config

def download_uber_data():
    if not os.path.exists(config.UBER_DATA_PATH):
        print("Downloading Uber raw dataset...")
        url = "https://raw.githubusercontent.com/fivethirtyeight/uber-tlc-foil-response/master/uber-trip-data/uber-raw-data-apr14.csv"
        # Download first 25,000 rows
        df_raw = pd.read_csv(url, nrows=25000)
        # Select Lat/Lon and sample 5,000 records
        df_sampled = df_raw[["Lat", "Lon"]].sample(n=5000, random_state=42).reset_index(drop=True)
        df_sampled.to_csv(config.UBER_DATA_PATH, index=False)
        print("Uber dataset downloaded and sampled.")
    else:
        print("Uber dataset already exists.")

def load_uber_data():
    download_uber_data()
    return pd.read_csv(config.UBER_DATA_PATH)
