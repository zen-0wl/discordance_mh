import json
import pickle

from tqdm import tqdm

from src.preprocessing.text.clean import clean_text

from src.config.settings import (
    ERISK_DATASET_ROOT,
    PROCESSED_DATA_DIR,
)

RAW_DIR = ERISK_DATASET_ROOT / "all_combined"

PROCESSED_DIR = PROCESSED_DATA_DIR

LABEL_FILE = ERISK_DATASET_ROOT / "shuffled_ground_truth_labels.txt"

OUTPUT_FILE = PROCESSED_DIR / "aggregated_users.pkl"

# ----------------------------------------------------
# labels
# ----------------------------------------------------

labels = {}

with open(LABEL_FILE, encoding="utf-8") as f:

    for line in f:

        subject, label = line.strip().split()

        labels[subject] = int(label)


# ----------------------------------------------------
# aggregate
# ----------------------------------------------------

dataset = []

json_files = sorted(RAW_DIR.glob("*.json"))

for file in tqdm(json_files):

    subject_id = file.stem

    if subject_id not in labels:
        continue

    with open(file, encoding="utf-8") as f:

        timeline = json.load(f)

    posts = []

    total_words = 0

    submission_count = len(timeline)

    for event in timeline:

        submission = event["submission"]

        title = clean_text(submission.get("title", ""))

        body = clean_text(submission.get("body", ""))

        document = []

        if title:
            document.append(title)

        if body:
            document.append(body)

        for comment in event.get("comments", []):

            if comment.get("target", False):

                text = clean_text(
                    comment.get("body", "")
                )

                if text:

                    document.append(text)

        document = "\n".join(document).strip()

        if len(document) == 0:
            continue

        posts.append(document)

        total_words += len(document.split())

    dataset.append({

        "subject_id": subject_id,

        "label": labels[subject_id],

        "posts": posts,

        "num_posts": len(posts),

        "num_submissions": submission_count,

        "word_count": total_words

    })


PROCESSED_DIR.mkdir(exist_ok=True)

with open(OUTPUT_FILE, "wb") as f:

    pickle.dump(dataset, f)

print(f"Saved {len(dataset)} users.")