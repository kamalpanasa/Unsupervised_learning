import pandas as pd

def clean_heroes_data(df):
    """
    Cleans superhero stats.
    """
    df_cleaned = df.dropna(subset=["Name", "Alignment"]).copy()
    # Ensure alignments are lowercase and standardized
    df_cleaned["Alignment"] = df_cleaned["Alignment"].str.lower()
    return df_cleaned
