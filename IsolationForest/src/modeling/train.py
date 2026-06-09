import os
import sys
import pickle
import numpy as np
from sklearn.ensemble import IsolationForest

# Add root directory to python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils import config
from src.data.load_data import load_turbofan_data
from src.data.clean_data import clean_turbofan_data
from src.features.build_features import scale_sensors, get_non_constant_sensors

def run_training_pipeline(contamination=0.10):
    # 1. Load data
    df_raw = load_turbofan_data()
    
    # 2. Clean data
    df_cleaned = clean_turbofan_data(df_raw)
    
    # 3. Identify non-constant sensors
    sensor_cols = get_non_constant_sensors(df_cleaned)
    
    # 4. Scale features
    X_scaled, scaler = scale_sensors(df_cleaned, sensor_cols)
    
    # 5. Fit Isolation Forest
    print(f"Fitting Isolation Forest (contamination={contamination})...")
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    anomaly_labels = iso_forest.fit_predict(X_scaled)
    
    # Add predictions
    df_cleaned["anomaly"] = np.where(anomaly_labels == -1, 1, 0)
    df_cleaned["anomaly_score"] = -iso_forest.decision_function(X_scaled) # higher score = more anomalous
    
    # Save output data
    df_cleaned.to_csv(config.CLUSTERED_TURBOFAN_PATH, index=False)
    
    # Save model metadata
    models = {
        "scaler": scaler,
        "iso_forest": iso_forest,
        "sensor_cols": sensor_cols,
        "contamination": contamination
    }
    with open(config.MODELS_PATH, "wb") as f:
        pickle.dump(models, f)
        
    print("Isolation Forest training complete. Saved models.pkl and turbofan_clustered.csv.")

if __name__ == "__main__":
    run_training_pipeline()
