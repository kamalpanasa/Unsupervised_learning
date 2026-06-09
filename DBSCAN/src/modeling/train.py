import os
import sys
import pickle
import numpy as np
from sklearn.cluster import DBSCAN

# Add root directory to python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils import config
from src.data.load_data import load_uber_data
from src.data.clean_data import clean_uber_data
from src.features.build_features import scale_coordinates

def run_training_pipeline(eps=0.15, min_samples=10):
    # 1. Load data
    df_raw = load_uber_data()
    
    # 2. Clean data
    df_cleaned = clean_uber_data(df_raw)
    
    # 3. Scale features
    X_scaled, scaler = scale_coordinates(df_cleaned)
    
    # 4. Fit DBSCAN
    print(f"Fitting DBSCAN (eps={eps}, min_samples={min_samples})...")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)
    
    df_cleaned["cluster"] = labels
    
    # Save output data
    df_cleaned.to_csv(config.CLUSTERED_UBER_PATH, index=False)
    
    # Save model metadata
    models = {
        "scaler": scaler,
        "eps": eps,
        "min_samples": min_samples,
        "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
        "n_noise": list(labels).count(-1)
    }
    with open(config.MODELS_PATH, "wb") as f:
        pickle.dump(models, f)
        
    print(f"DBSCAN training complete. Found {models['n_clusters']} clusters and {models['n_noise']} noise points.")

if __name__ == "__main__":
    run_training_pipeline()
