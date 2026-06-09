import os
import sys
import pickle

# Add root directory to python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage
from src.utils import config
from src.data.load_data import load_raw_data
from src.data.clean_data import clean_pokemon_data
from src.features.build_features import scale_features

def run_training_pipeline(n_clusters=5):
    # 1. Load raw data
    df_raw = load_raw_data()
    
    # 2. Clean data
    df_cleaned = clean_pokemon_data(df_raw)
    
    # 3. Scale features
    X_scaled, scaler = scale_features(df_cleaned)
    
    # 4. Linkage Matrix
    print("Computing linkage matrix...")
    linkage_matrix = linkage(X_scaled, method="ward")
    
    # 5. Fit Model
    print(f"Fitting Agglomerative Clustering (n_clusters={n_clusters})...")
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = model.fit_predict(X_scaled)
    
    # Add cluster labels
    df_cleaned["cluster"] = labels
    
    # Save outputs
    df_cleaned.to_csv(config.CLUSTERED_POKEMON_PATH, index=False)
    
    models = {
        "scaler": scaler,
        "linkage_matrix": linkage_matrix,
        "n_clusters": n_clusters,
        "stat_cols": config.STAT_COLS
    }
    with open(config.MODELS_PATH, "wb") as f:
        pickle.dump(models, f)
        
    print(f"Agglomerative model training complete. Saved models.pkl and pokemon_clustered.csv.")

if __name__ == "__main__":
    run_training_pipeline()
