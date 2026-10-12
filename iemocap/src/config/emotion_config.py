"""
Emotion label configuration for IEMOCAP.
"""

EMOTION_TO_ID = {

    "ang": 0,
    "exc": 1,
    "fru": 2,
    "hap": 3,
    "neu": 4,
    "sad": 5,

}

ID_TO_EMOTION = {

    value: key

    for key, value in EMOTION_TO_ID.items()

}

NUM_CLASSES = len(
    EMOTION_TO_ID
)

SUPPORTED_EMOTIONS = list(
    EMOTION_TO_ID.keys()
)