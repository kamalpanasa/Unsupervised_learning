from sklearn.preprocessing import StandardScaler
from src.utils import config

def scale_mall_features(df):
    """
    Standardizes Age, Annual Income, and Spending Score.
    """
    X = df[config.NUMERICAL_COLS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
