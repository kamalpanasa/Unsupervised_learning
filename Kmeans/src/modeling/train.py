import os
import sys
import pickle
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Add root directory to python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils import config
from src.data.load_data import load_mall_data
from src.data.clean_data import clean_mall_data
from src.features.build_features import scale_mall_features

def run_training_pipeline(k=5):
    # 1. Load data
    df_raw = load_mall_data()
    
    # 2. Clean data
    df_cleaned = clean_mall_data(df_raw)
    
    # 3. Scale features
    X_scaled, scaler = scale_mall_features(df_cleaned)
    
    # 4. Precompute Elbow metrics (WCSS and Silhouette) for K=2..8
    print("Computing elbow curve metrics...")
    k_range = list(range(2, 9))
    wcss = []
    silhouette_scores = []
    
    for val in k_range:
        km = KMeans(n_clusters=val, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        wcss.append(float(km.inertia_))
        silhouette_scores.append(float(silhouette_score(X_scaled, labels)))
        
    elbow_metrics = {
        "k_values": k_range,
        "wcss": wcss,
        "silhouette_scores": silhouette_scores
    }
    with open(config.ELBOW_METRICS_PATH, "w") as f:
        json.dump(elbow_metrics, f)
        
    # 5. Fit target K-Means model
    print(f"Fitting KMeans (K={k})...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    df_cleaned["cluster"] = labels
    
    # Save output data
    df_cleaned.to_csv(config.CLUSTERED_MALL_PATH, index=False)
    
    # Save model metadata
    models = {
        "scaler": scaler,
        "kmeans": kmeans,
        "n_clusters": k,
        "numerical_cols": config.NUMERICAL_COLS
    }
    with open(config.MODELS_PATH, "wb") as f:
        pickle.dump(models, f)
        
    print("KMeans training complete. Saved models.pkl and mall_customers_clustered.csv.")

if __name__ == "__main__":
    run_training_pipeline()
