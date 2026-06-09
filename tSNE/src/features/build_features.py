import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.utils import config

def build_tsne_features(df):
    """
    Performs one-hot encoding on categorical variables and standardizes features.
    """
    df_features = pd.get_dummies(df[config.CAT_COLS + config.GEOM_COLS])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features.values)
    
    return X_scaled, scaler, list(df_features.columns)
