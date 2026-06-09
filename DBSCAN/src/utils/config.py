import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# File Paths
UBER_DATA_PATH = os.path.join(DATA_DIR, "uber_pickups.csv")
CLUSTERED_UBER_PATH = os.path.join(DATA_DIR, "uber_pickups_clustered.csv")
MODELS_PATH = os.path.join(MODEL_DIR, "models.pkl")

# Columns
COORDS_COLS = ["Lat", "Lon"]
