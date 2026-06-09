import os
import sys
import pickle
from sklearn.mixture import GaussianMixture

# Add root directory to python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils import config
from src.data.load_data import load_spotify_data
from src.data.clean_data import clean_spotify_data
from src.features.build_features import scale_audio_features

def run_training_pipeline(n_components=4):
    # 1. Load data
    df_raw = load_spotify_data()
    
    # 2. Clean data
    df_cleaned = clean_spotify_data(df_raw)
    
    # 3. Scale features
    X_scaled, scaler = scale_audio_features(df_cleaned)
    
    # 4. Fit GMM
    print(f"Fitting Gaussian Mixture Model (n_components={n_components})...")
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X_scaled)
    
    # Soft & Hard cluster labels
    probs = gmm.predict_proba(X_scaled)
    labels = gmm.predict(X_scaled)
    
    # Add to dataframe
    df_cleaned["cluster"] = labels
    for i in range(n_components):
        df_cleaned[f"prob_cluster_{i}"] = probs[:, i]
        
    # Save clustered data
    df_cleaned.to_csv(config.CLUSTERED_SPOTIFY_PATH, index=False)
    
    # Save model pickle
    models = {
        "scaler": scaler,
        "gmm": gmm,
        "n_components": n_components,
        "audio_features": config.AUDIO_FEATURES
    }
    with open(config.MODELS_PATH, "wb") as f:
        pickle.dump(models, f)
        
    print("GMM model training complete. Models saved.")

if __name__ == "__main__":
    run_training_pipeline()
