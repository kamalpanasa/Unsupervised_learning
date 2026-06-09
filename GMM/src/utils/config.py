import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# File Paths
SPOTIFY_DATA_PATH = os.path.join(DATA_DIR, "spotify_songs.csv")
CLUSTERED_SPOTIFY_PATH = os.path.join(DATA_DIR, "spotify_clustered.csv")
MODELS_PATH = os.path.join(MODEL_DIR, "models.pkl")

# Constant column lists
AUDIO_FEATURES = ["danceability", "energy", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"]
