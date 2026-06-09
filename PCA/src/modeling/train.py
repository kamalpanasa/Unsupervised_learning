import os
import sys
import pickle
from sklearn.decomposition import PCA

# Add root directory to python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils import config
from src.data.load_data import load_heroes_data
from src.data.clean_data import clean_heroes_data
from src.features.build_features import scale_power_stats

def run_training_pipeline():
    # 1. Load data
    df_raw = load_heroes_data()
    
    # 2. Clean data
    df_cleaned = clean_heroes_data(df_raw)
    
    # 3. Scale features
    X_scaled, scaler = scale_power_stats(df_cleaned)
    
    # 4. Fit PCA (3 components)
    print("Fitting PCA model...")
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    
    # Add PCs to dataframe
    df_cleaned["PC1"] = X_pca[:, 0]
    df_cleaned["PC2"] = X_pca[:, 1]
    df_cleaned["PC3"] = X_pca[:, 2]
    
    # Save output data
    df_cleaned.to_csv(config.CLUSTERED_HEROES_PATH, index=False)
    
    # Save model pickle
    models = {
        "scaler": scaler,
        "pca": pca,
        "stat_cols": config.STAT_COLS,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "components": pca.components_
    }
    with open(config.MODELS_PATH, "wb") as f:
        pickle.dump(models, f)
        
    print("PCA training complete. Saved models.pkl and superheros_pca.csv.")

if __name__ == "__main__":
    run_training_pipeline()
