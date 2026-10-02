"""
data_loader.py
--------------
Dataset handling: loads and preprocesses the Crop Recommendation Dataset.
Source: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
"""
import os
import pandas as pd

DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset", "Crop_recommendation.csv")

# The 7 input features used by the model (order matters!)
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"


def load_dataset(path=DATASET_PATH):
    """Read the CSV file into a Pandas DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download 'Crop_recommendation.csv' from Kaggle and place it in the 'dataset' folder."
        )
    return pd.read_csv(path)


def preprocess(df):
    """Basic preprocessing: drop missing values and duplicates, split into X and y."""
    df = df.dropna().drop_duplicates()
    X = df[FEATURES]          # input features
    y = df[TARGET]            # crop name (what we want to predict)
    return X, y
