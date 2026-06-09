from sklearn.preprocessing import StandardScaler
from src.utils import config

def scale_audio_features(df):
    """
    Standardizes Spotify audio columns.
    """
    X = df[config.AUDIO_FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
