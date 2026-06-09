import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# File Paths
HEROES_DATA_PATH = os.path.join(DATA_DIR, "superheros.csv")
CLUSTERED_HEROES_PATH = os.path.join(DATA_DIR, "superheros_pca.csv")
MODELS_PATH = os.path.join(MODEL_DIR, "models.pkl")

# Constant column lists
STAT_COLS = ["Intelligence", "Strength", "Speed", "Durability", "Power", "Combat"]
