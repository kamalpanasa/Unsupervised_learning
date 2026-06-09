import pandas as pd
from src.utils import config

def clean_pokemon_data(df):
    """
    Cleans the Pokemon dataset, drops null values in stats,
    and renames 'Pikachu' to 'vilohih'.
    """
    df_cleaned = df.dropna(subset=config.STAT_COLS + ["Name"]).copy()
    
    # Rename Pikachu to vilohih
    df_cleaned["Name"] = df_cleaned["Name"].replace("Pikachu", "vilohih")
    
    return df_cleaned
