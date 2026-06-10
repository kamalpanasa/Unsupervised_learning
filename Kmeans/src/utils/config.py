import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DATA_DIR = os.path.join(ROOT_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(ROOT_DIR, "models")

# Ensure directories exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# File Paths
MALL_DATA_PATH = os.path.join(DATA_DIR, "mall_customers.csv")
CLUSTERED_MALL_PATH = os.path.join(PROCESSED_DIR, "mall_customers_clustered.csv")
ELBOW_METRICS_PATH = os.path.join(PROCESSED_DIR, "elbow_metrics.json")
MODELS_PATH = os.path.join(MODEL_DIR, "models.pkl")

# Constant column lists
NUMERICAL_COLS = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
