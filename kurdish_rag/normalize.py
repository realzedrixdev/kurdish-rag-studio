import re
import unicodedata

MAP = str.maketrans({"ك": "ک", "ي": "ی", "ى": "ی", "ة": "ە", "ۀ": "ە", "ؤ": "ۆ", "إ": "ئ", "أ": "ئ"})
INVISIBLE = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u200e\u200f\ufeff"), None)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(MAP).translate(INVISIBLE)
    return re.sub(r"\s+", " ", text).strip().casefold()


def tokens(text: str) -> list[str]:
    return re.findall(r"[\w\u0600-\u06ff]+", normalize(text), flags=re.UNICODE)


def trigrams(text: str) -> set[str]:
    compact = f"  {normalize(text)}  "
    return {compact[i:i + 3] for i in range(max(0, len(compact) - 2))}
