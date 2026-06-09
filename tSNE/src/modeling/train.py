import os
import sys
import pickle
from sklearn.manifold import TSNE

# Add root directory to python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils import config
from src.data.load_data import load_anime_data
from src.data.clean_data import clean_anime_data
from src.features.build_features import build_tsne_features

def run_training_pipeline(perplexity=30, learning_rate="auto"):
    # 1. Load data
    df_raw = load_anime_data()
    
    # 2. Clean data
    df_cleaned = clean_anime_data(df_raw)
    
    # 3. Scale & One-hot features
    X_scaled, scaler, feature_cols = build_tsne_features(df_cleaned)
    
    # 4. Fit t-SNE
    print(f"Running t-SNE (perplexity={perplexity}, learning_rate={learning_rate})...")
    tsne = TSNE(n_components=3, perplexity=perplexity, learning_rate=learning_rate, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    
    # Save coordinate projections
    df_cleaned["tSNE_1"] = X_tsne[:, 0]
    df_cleaned["tSNE_2"] = X_tsne[:, 1]
    df_cleaned["tSNE_3"] = X_tsne[:, 2]
    
    df_cleaned.to_csv(config.CLUSTERED_ANIME_PATH, index=False)
    
    # Save model metadata
    models = {
        "perplexity": perplexity,
        "learning_rate": learning_rate,
        "feature_cols": feature_cols
    }
    with open(config.MODELS_PATH, "wb") as f:
        pickle.dump(models, f)
        
    print("t-SNE pipeline training complete. Coordinates saved.")

if __name__ == "__main__":
    run_training_pipeline()
