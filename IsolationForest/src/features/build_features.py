from sklearn.preprocessing import StandardScaler
from src.utils import config

def get_non_constant_sensors(df):
    """
    Finds sensor columns that show variance (nunique > 1).
    """
    return [col for col in config.SENSOR_COLS if df[col].nunique() > 1]

def scale_sensors(df, sensor_cols):
    """
    Standardizes sensor telemetry readings.
    """
    X = df[sensor_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
