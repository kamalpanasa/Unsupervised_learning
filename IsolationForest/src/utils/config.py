import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# File Paths
TURBOFAN_DATA_PATH = os.path.join(DATA_DIR, "turbofan_sensors.csv")
CLUSTERED_TURBOFAN_PATH = os.path.join(DATA_DIR, "turbofan_clustered.csv")
MODELS_PATH = os.path.join(MODEL_DIR, "models.pkl")

# Constant column groupings
META_COLS = ["unit_number", "cycle"]
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
