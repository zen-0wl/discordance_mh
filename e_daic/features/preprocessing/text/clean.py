import re


def clean_text(text: str) -> str:
    """
    Basic transcript cleaning for E-DAIC transcripts.
    Keeps semantic meaning intact while normalizing formatting.
    """

    if not isinstance(text, str):
        return ""

    # lowercase
    text = text.lower()

    # remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    # remove repeated punctuation
    text = re.sub(r"[.]{2,}", ".", text)
    text = re.sub(r"[!]{2,}", "!", text)
    text = re.sub(r"[?]{2,}", "?", text)

    # strip surrounding spaces
    text = text.strip()

    return text


def is_valid_text(text: str, min_words: int = 1) -> bool:
    """
    Validate transcript text quality.
    """

    if not text:
        return False

    words = text.split()

    if len(words) < min_words:
        return False

    return True

# heuristic patterns to identify interviewer speech
INTERVIEWER_PATTERNS = [
    "how are you",
    "can you",
    "tell me",
    "why do you",
    "what do you",
    "where are you from",
    "could you",
    "would you",
    "do you feel",
    "do you think",
]

def is_likely_interviewer(text: str) -> bool:

    text = text.lower().strip()

    for pattern in INTERVIEWER_PATTERNS:
        if pattern in text:
            return True

    return False