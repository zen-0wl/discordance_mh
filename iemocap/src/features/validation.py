"""
Checks:

✓ Dataset size
✓ Missing values
✓ Embedding dimensions
✓ NaNs
✓ Infinite values
✓ Zero vectors
✓ Embedding norms
✓ Label distribution
✓ Duration statistics
"""

import numpy as np
from collections import Counter


def validate_dataset(dataset):

    print("=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)

    validate_size(dataset)
    validate_shapes(dataset)
    validate_nan(dataset)
    validate_inf(dataset)
    validate_zero_vectors(dataset)
    validate_norms(dataset)
    validate_labels(dataset)
    validate_duration(dataset)

    print()
    print("=" * 60)
    print("ALL VALIDATION CHECKS PASSED")
    print("=" * 60)


# ---------------------------------------------------------


def validate_size(dataset):

    print()

    print(f"Dataset size : {len(dataset)}")


# ---------------------------------------------------------


def validate_shapes(dataset):

    audio_shapes = {
        sample["audio_embedding"].shape
        for sample in dataset
    }

    text_shapes = {
        sample["text_embedding"].shape
        for sample in dataset
    }

    print()

    print("Audio shapes :", audio_shapes)
    print("Text shapes  :", text_shapes)

    assert audio_shapes == {(768,)}

    assert text_shapes == {(768,)}


# ---------------------------------------------------------


def validate_nan(dataset):

    audio_nan = 0

    text_nan = 0

    for sample in dataset:

        if np.isnan(
            sample["audio_embedding"]
        ).any():

            audio_nan += 1

        if np.isnan(
            sample["text_embedding"]
        ).any():

            text_nan += 1

    print()

    print("Audio NaNs :", audio_nan)

    print("Text NaNs  :", text_nan)

    assert audio_nan == 0

    assert text_nan == 0


# ---------------------------------------------------------


def validate_inf(dataset):

    audio_inf = 0

    text_inf = 0

    for sample in dataset:

        if np.isinf(
            sample["audio_embedding"]
        ).any():

            audio_inf += 1

        if np.isinf(
            sample["text_embedding"]
        ).any():

            text_inf += 1

    print()

    print("Audio Inf :", audio_inf)

    print("Text Inf  :", text_inf)

    assert audio_inf == 0

    assert text_inf == 0


# ---------------------------------------------------------


def validate_zero_vectors(dataset):

    audio_zero = 0

    text_zero = 0

    for sample in dataset:

        if np.allclose(
            sample["audio_embedding"],
            0,
        ):

            audio_zero += 1

        if np.allclose(
            sample["text_embedding"],
            0,
        ):

            text_zero += 1

    print()

    print("Zero Audio :", audio_zero)

    print("Zero Text  :", text_zero)

    assert audio_zero == 0

    assert text_zero == 0


# ---------------------------------------------------------


def validate_norms(dataset):

    audio_norms = [

        np.linalg.norm(
            sample["audio_embedding"]
        )

        for sample in dataset

    ]

    text_norms = [

        np.linalg.norm(
            sample["text_embedding"]
        )

        for sample in dataset

    ]

    print()

    print(
        "Average Audio Norm :",
        np.mean(audio_norms),
    )

    print(
        "Average Text Norm  :",
        np.mean(text_norms),
    )

    print()

    print(
        "Audio Norm Range :",

        round(min(audio_norms), 3),

        "-",

        round(max(audio_norms), 3),

    )

    print(
        "Text Norm Range  :",

        round(min(text_norms), 3),

        "-",

        round(max(text_norms), 3),

    )


# ---------------------------------------------------------


def validate_labels(dataset):

    labels = [

        sample["emotion"]

        for sample in dataset

    ]

    counts = Counter(labels)

    print()

    print("Emotion Distribution")

    print("--------------------")

    for emotion, count in sorted(counts.items()):

        print(

            f"{emotion:>5} : {count}"

        )


# ---------------------------------------------------------


def validate_duration(dataset):

    durations = [

        sample["duration"]

        for sample in dataset

    ]

    print()

    print("Duration Statistics")

    print("-------------------")

    print(

        "Mean :",

        round(np.mean(durations), 3),

    )

    print(

        "Median :",

        round(np.median(durations), 3),

    )

    print(

        "Minimum :",

        round(min(durations), 3),

    )

    print(

        "Maximum :",

        round(max(durations), 3),

    )


# ---------------------------------------------------------


def dataset_summary(dataset):

    print()

    print("=" * 60)

    print("DATASET SUMMARY")

    print("=" * 60)

    sample = dataset[0]

    print()

    print("Keys")

    print(sample.keys())

    print()

    print(

        "Audio Shape:",

        sample["audio_embedding"].shape,

    )

    print(

        "Text Shape:",

        sample["text_embedding"].shape,

    )

    print()

    print(

        "Example Emotion:",

        sample["emotion"],

    )

    print()

    print(

        "Example Transcript:",

        sample["transcript"],

    )

    print()