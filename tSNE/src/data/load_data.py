import os
import pandas as pd
import numpy as np
from src.utils import config

def generate_anime_dataset(n_samples_per_group=200):
    np.random.seed(42)
    groups = ["Chibi", "Kuudere", "Shonen Protagonist", "Tsundere", "Villain"]
    data = []
    char_id = 1
    
    for g in groups:
        for _ in range(n_samples_per_group):
            if g == "Chibi":
                jaw_width = np.random.normal(40, 5)
                eye_size = np.random.normal(85, 4)
                smile_score = np.random.normal(90, 5)
                blush_intensity = np.random.normal(75, 8)
                accessory_count = np.random.normal(3, 1)
                hair_color = "Pink"
                eye_color = "Blue"
            elif g == "Kuudere":
                jaw_width = np.random.normal(50, 4)
                eye_size = np.random.normal(55, 5)
                smile_score = np.random.normal(10, 3)
                blush_intensity = np.random.normal(15, 4)
                accessory_count = np.random.normal(0, 0.5)
                hair_color = "Blue"
                eye_color = "Gray"
            elif g == "Shonen Protagonist":
                jaw_width = np.random.normal(60, 5)
                eye_size = np.random.normal(70, 5)
                smile_score = np.random.normal(60, 8)
                blush_intensity = np.random.normal(45, 10)
                accessory_count = np.random.normal(1.5, 0.8)
                hair_color = "Black"
                eye_color = "Brown"
            elif g == "Tsundere":
                jaw_width = np.random.normal(45, 4)
                eye_size = np.random.normal(75, 4)
                smile_score = np.random.normal(25, 6)
                blush_intensity = np.random.normal(90, 5)
                accessory_count = np.random.normal(2, 0.7)
                hair_color = "Blonde"
                eye_color = "Red"
            elif g == "Villain":
                jaw_width = np.random.normal(65, 6)
                eye_size = np.random.normal(40, 5)
                smile_score = np.random.normal(30, 10)
                blush_intensity = np.random.normal(5, 2)
                accessory_count = np.random.normal(1, 0.8)
                hair_color = "Purple"
                eye_color = "Yellow"
                
            data.append({
                "Character_ID": f"Char_{char_id:04d}",
                "Style_Group": g,
                "Hair_Color": hair_color,
                "Eye_Color": eye_color,
                "Jaw_Width": max(10, min(100, jaw_width)),
                "Eye_Size": max(10, min(100, eye_size)),
                "Smile_Score": max(0, min(100, smile_score)),
                "Blush_Intensity": max(0, min(100, blush_intensity)),
                "Accessory_Count": max(0, accessory_count)
            })
            char_id += 1
            
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

def load_anime_data():
    if not os.path.exists(config.ANIME_DATA_PATH):
        print("Generating synthetic Anime Face Features dataset...")
        df = generate_anime_dataset()
        df.to_csv(config.ANIME_DATA_PATH, index=False)
        print("Dataset generated and saved.")
    else:
        df = pd.read_csv(config.ANIME_DATA_PATH)
    return df
