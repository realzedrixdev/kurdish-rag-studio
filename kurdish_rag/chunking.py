from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    ordinal: int
    text: str


def chunk_text(text: str, max_chars: int = 900, overlap_chars: int = 120) -> list[Chunk]:
    paragraphs = [part.strip() for part in text.replace("\r\n", "\n").split("\n\n") if part.strip()]
    chunks: list[Chunk] = []
    current = ""
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) + 2 > max_chars:
            chunks.append(Chunk(len(chunks), current))
            tail = current[-overlap_chars:].lstrip() if overlap_chars else ""
            current = f"{tail}\n\n{paragraph}".strip()
        else:
            current = f"{current}\n\n{paragraph}".strip()
        while len(current) > max_chars:
            cut = current.rfind(" ", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars
            chunks.append(Chunk(len(chunks), current[:cut].strip()))
            current = current[max(0, cut - overlap_chars):].strip()
    if current:
        chunks.append(Chunk(len(chunks), current))
    return chunks
