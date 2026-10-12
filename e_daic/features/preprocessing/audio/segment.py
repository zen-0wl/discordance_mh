from pathlib import Path
from typing import List

import librosa
import pandas as pd

from src.dataset.schema import UtteranceSample
from src.preprocessing.text.clean import clean_text, is_valid_text, is_likely_interviewer

# preprocessing thresholds
MIN_DURATION = 0.5
MAX_DURATION = 20.0

MIN_CONFIDENCE = 0.60


def load_audio(audio_path: str):

    audio, sr = librosa.load(audio_path, sr=None)

    return audio, sr


def segment_audio(
    audio_path: str,
    transcript_df: pd.DataFrame,
    participant_id: str,
) -> List[UtteranceSample]:
    """
    segment full-session audio into utterance-level samples
    using transcript timestamps
    """

    audio, sr = load_audio(audio_path)

    utterances = []

    for idx, row in transcript_df.iterrows():

        start_time = float(row["Start_Time"])
        end_time = float(row["End_Time"])

        duration = end_time - start_time

        text = clean_text(row["Text"])
        confidence = float(row["Confidence"])

        # ---------------------------------------------------
        # filtering
        # ---------------------------------------------------

        if duration < MIN_DURATION:
            continue

        if duration > MAX_DURATION:
            continue

        if confidence < MIN_CONFIDENCE:
            continue

        if not is_valid_text(text):
            continue
        
        if is_likely_interviewer(text):
            continue

        # ---------------------------------------------------
        # convert timestamps to audio samples
        # ---------------------------------------------------
        PADDING = 0.15  # seconds

        start_time = max(0, start_time - PADDING)
        end_time = min(len(audio) / sr, end_time + PADDING)
        
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        segment = audio[start_sample:end_sample]

        utterance = UtteranceSample(
            participant_id=participant_id,
            utterance_id=f"{participant_id}_{idx}",
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            text=text,
            confidence=confidence,
            audio=segment,
            sample_rate=sr,
        )

        utterances.append(utterance)

    return utterances