import os
import pandas as pd
from src.utils import config

def download_turbofan_data():
    if not os.path.exists(config.TURBOFAN_DATA_PATH):
        print("Downloading NASA Turbofan Engine dataset...")
        # Note the verified url from mapr-demos
        url = "https://raw.githubusercontent.com/mapr-demos/predictive-maintenance/master/notebooks/jupyter/Dataset/CMAPSSData/train_FD001.txt"
        
        # Space-separated values
        col_names = config.META_COLS + config.SETTING_COLS + config.SENSOR_COLS
        df_raw = pd.read_csv(url, sep=r"\s+", header=None, names=col_names)
        
        # Filter first 10 engines to keep visualisations snappy and clean
        df_filtered = df_raw[df_raw["unit_number"] <= 10].reset_index(drop=True)
        df_filtered.to_csv(config.TURBOFAN_DATA_PATH, index=False)
        print("NASA Turbofan dataset downloaded and filtered.")
    else:
        print("NASA Turbofan dataset already exists.")

def load_turbofan_data():
    download_turbofan_data()
    return pd.read_csv(config.TURBOFAN_DATA_PATH)
