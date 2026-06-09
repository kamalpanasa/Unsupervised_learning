import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# File Paths
ANIME_DATA_PATH = os.path.join(DATA_DIR, "anime_faces.csv")
CLUSTERED_ANIME_PATH = os.path.join(DATA_DIR, "anime_faces_tsne.csv")
MODELS_PATH = os.path.join(MODEL_DIR, "models.pkl")

# Constant lists
GEOM_COLS = ["Jaw_Width", "Eye_Size", "Smile_Score", "Blush_Intensity", "Accessory_Count"]
CAT_COLS = ["Hair_Color", "Eye_Color"]
