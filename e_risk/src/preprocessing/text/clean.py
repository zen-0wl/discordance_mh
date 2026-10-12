import re


URL_PATTERN = re.compile(r"http\S+|www\.\S+")
WHITESPACE_PATTERN = re.compile(r"\s+")


def remove_urls(text: str) -> str:

    return URL_PATTERN.sub("", text)


def remove_deleted(text: str) -> str:

    text = text.replace("[deleted]", "")
    text = text.replace("[removed]", "")

    return text


def normalize_whitespace(text: str) -> str:

    return WHITESPACE_PATTERN.sub(" ", text).strip()


def clean_text(text: str) -> str:

    if text is None:
        return ""

    text = str(text)

    text = remove_urls(text)

    text = remove_deleted(text)

    text = normalize_whitespace(text)

    return text