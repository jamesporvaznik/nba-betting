import pandas as pd
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_data():
    path = "data/processed/games.csv"
    if not os.path.exists(path):
        raise FileNotFoundError("No data found. Run the scrapers first.")
    return pd.read_csv(path)

def train():
    df = load_data()
    print(f"Loaded {len(df)} rows")
    # Your feature engineering and training goes here
    print("Add your feature engineering and model training here")

if __name__ == "__main__":
    train()