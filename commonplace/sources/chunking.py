import re

_WS = re.compile(r"\s")


def _window_end(text: str, start: int, size: int) -> int:
    """Where a chunk starting at `start` ends: at most `size` chars on, cut back to whitespace."""
    end = min(start + size, len(text))
    if end < len(text):
        cut = max((m.start() for m in _WS.finditer(text, start + 1, end)), default=-1)
        if cut > start:
            end = cut
        # ponytail: a single word longer than `size` gets hard-cut; real prose never has one
    return end


def _skip_whitespace(text: str, position: int) -> int:
    while position < len(text) and text[position].isspace():
        position += 1
    return position


def chunk_text(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into windows of at most `chunk_size` chars that overlap by about `chunk_overlap`.

    Cuts only on whitespace, so no word is split. Each window after the first
    starts at the first word boundary inside the last `chunk_overlap` chars of the
    previous one, so an idea sitting on a boundary is whole in at least one chunk.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    text = text.strip()

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = _window_end(text, start, chunk_size)
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        boundary = _WS.search(text, max(end - chunk_overlap, start + 1), end)
        overlapped = _skip_whitespace(text, boundary.end() if boundary else end)
        # When the next word is too long to fit after the overlap, a window that
        # starts inside the overlap would end where this one did and repeat it.
        if _window_end(text, overlapped, chunk_size) <= end:
            overlapped = _skip_whitespace(text, end)
        start = overlapped
    return chunks
