from sklearn.preprocessing import StandardScaler
from src.utils import config

def scale_features(df):
    """
    Extracts combat stats and standardizes them.
    """
    X = df[config.STAT_COLS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
