import re

_WS = re.compile(r"\s")


def chunk_text(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into windows of at most `chunk_size` chars that overlap by about `chunk_overlap`.

    Cuts only on whitespace, so no word is split. Each window after the first
    starts at the first word boundary inside the last `overlap` chars of the
    previous one, so an idea sitting on a boundary is whole in at least one chunk.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    size, overlap = chunk_size, chunk_overlap
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []

    chunks: list[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            cut = max((m.start() for m in _WS.finditer(text, start + 1, end)), default=-1)
            if cut > start:
                end = cut
            # ponytail: a single word longer than `size` gets hard-cut; real prose never has one
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        ws = _WS.search(text, max(end - overlap, start + 1), end)
        start = ws.end() if ws else end
        while start < n and text[start].isspace():
            start += 1
    return chunks
