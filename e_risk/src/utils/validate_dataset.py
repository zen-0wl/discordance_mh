import pickle
import numpy as np

from src.config.settings import (
    ERISK_FEATURE_ROOT,
)

# ---------------------------------------------------
# paths
# ---------------------------------------------------

EMBEDDING_FILE = (
    ERISK_FEATURE_ROOT /
    "embeddings.pkl"
)

# ---------------------------------------------------
# load
# ---------------------------------------------------

with open(EMBEDDING_FILE, "rb") as f:

    dataset = pickle.load(f)

print("=" * 60)
print("eRisk Embedding Validation")
print("=" * 60)

print()

print(f"Subjects: {len(dataset)}")

# ---------------------------------------------------
# labels
# ---------------------------------------------------

labels = [sample["label"] for sample in dataset]

control = sum(label == 0 for label in labels)

depressed = sum(label == 1 for label in labels)

print()
print("Label Distribution")
print("-" * 30)

print(f"Control   : {control}")
print(f"Depressed : {depressed}")

# ---------------------------------------------------
# sequence lengths
# ---------------------------------------------------

sequence_lengths = [

    sample["sequence_length"]

    for sample in dataset

]

print()
print("Embedding Sequence Length")
print("-" * 30)

print(f"Average : {np.mean(sequence_lengths):.2f}")
print(f"Median  : {np.median(sequence_lengths):.2f}")
print(f"Maximum : {np.max(sequence_lengths)}")
print(f"Minimum : {np.min(sequence_lengths)}")

# ---------------------------------------------------
# embedding dimensions
# ---------------------------------------------------

dimensions = [

    sample["embedding_sequence"].shape[1]

    for sample in dataset

]

print()
print("Embedding Dimension")
print("-" * 30)

print(f"Unique dimensions : {sorted(set(dimensions))}")

# ---------------------------------------------------
# dtype
# ---------------------------------------------------

dtypes = [

    sample["embedding_sequence"].dtype

    for sample in dataset

]

print()
print("Embedding dtype")
print("-" * 30)

print(set(dtypes))

# ---------------------------------------------------
# NaN / Inf checks
# ---------------------------------------------------

nan_count = 0

inf_count = 0

for sample in dataset:

    emb = sample["embedding_sequence"]

    nan_count += np.isnan(emb).sum()

    inf_count += np.isinf(emb).sum()

print()
print("Numerical Integrity")
print("-" * 30)

print(f"NaNs : {nan_count}")
print(f"Infs : {inf_count}")

# ---------------------------------------------------
# embedding statistics
# ---------------------------------------------------

all_embeddings = np.concatenate(

    [

        sample["embedding_sequence"]

        for sample in dataset

    ],

    axis=0,

)

print()
print("Embedding Statistics")
print("-" * 30)

print(f"Mean : {all_embeddings.mean():.6f}")
print(f"Std  : {all_embeddings.std():.6f}")
print(f"Min  : {all_embeddings.min():.6f}")
print(f"Max  : {all_embeddings.max():.6f}")

# ---------------------------------------------------
# first sample
# ---------------------------------------------------

example = dataset[0]

print()
print("Example Subject")
print("-" * 30)

print(f"Subject ID      : {example['subject_id']}")
print(f"Label           : {example['label']}")
print(f"Sequence Shape  : {example['embedding_sequence'].shape}")
print(f"Sequence Length : {example['sequence_length']}")

if "original_posts" in example:
    print(f"Original Posts  : {example['original_posts']}")

if "encoder" in example:
    print(f"Encoder         : {example['encoder']}")

# ---------------------------------------------------
# validation
# ---------------------------------------------------

passed = (

    len(dataset) == 909

    and control == 807

    and depressed == 102

    and nan_count == 0

    and inf_count == 0

    and set(dimensions) == {768}

)

print()

print("=" * 60)

if passed:

    print("Embedding validation PASSED")

else:

    print("Embedding validation FAILED")

print("=" * 60)