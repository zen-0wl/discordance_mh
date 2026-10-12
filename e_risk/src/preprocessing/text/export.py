from pathlib import Path
import pickle

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]

PROCESSED_DIR = ROOT / "data" / "processed"

INPUT_FILE = PROCESSED_DIR / "aggregated_users.pkl"

CSV_FILE = PROCESSED_DIR / "dataset.csv"

PKL_FILE = PROCESSED_DIR / "dataset.pkl"


with open(INPUT_FILE, "rb") as f:

    dataset = pickle.load(f)


rows = []

for sample in dataset:

    rows.append({

        "subject_id": sample["subject_id"],

        "label": sample["label"],

        "posts": sample["posts"],

        "num_posts": sample["num_posts"],

        "num_submissions": sample["num_submissions"],

        "word_count": sample["word_count"]

    })


df = pd.DataFrame(rows)

print(df.head())

print(df.shape)

df.to_csv(CSV_FILE, index=False)

df.to_pickle(PKL_FILE)

print("Dataset exported successfully.")