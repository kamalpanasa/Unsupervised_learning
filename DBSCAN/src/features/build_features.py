from sklearn.preprocessing import StandardScaler
from src.utils import config

def scale_coordinates(df):
    """
    Standardizes Lat and Lon coordinates.
    """
    X = df[config.COORDS_COLS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
