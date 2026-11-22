from datetime import UTC, datetime
from pathlib import Path

import pytest

from commonplace.sources.allowlist import NotAllowed, Publication, assert_allowed, find_by_feed
from commonplace.sources.chunk import chunk
from commonplace.sources.parse import extract_article
from commonplace.sources.rss import parse_feed
from commonplace.sources.urls import canonical_id

FIXTURES = Path(__file__).parent.parent / "fixtures"

PUBS = [
    Publication(
        slug="latent-space",
        name="Latent Space",
        feed_url="https://www.latent.space/feed",
        homepage="https://www.latent.space",
    ),
]


@pytest.mark.parametrize(
    "url",
    [
        "https://latent.space/p/formal-verification",
        "https://www.latent.space/p/formal-verification/",
        "http://LATENT.space/p/formal-verification?utm_source=x&r=1#comments",
    ],
)
def test_canonical_id_dedupes_tracking_params_and_slash(url: str) -> None:
    assert canonical_id(url) == "latent.space/p/formal-verification"


def test_allowlist_matches_feed_with_or_without_www() -> None:
    assert find_by_feed("https://latent.space/feed", PUBS) == PUBS[0]
    assert find_by_feed("https://evil.example/feed", PUBS) is None


def test_allowlist_rejects_unknown_host() -> None:
    assert assert_allowed("https://latent.space/p/anything", PUBS) == PUBS[0]
    with pytest.raises(NotAllowed):
        assert_allowed("https://evil.example/p/anything", PUBS)


def test_chunk_short_text_is_one_chunk() -> None:
    assert chunk("hello world", size=1000, overlap=150) == ["hello world"]
    assert chunk("   \n ", size=1000, overlap=150) == []


def test_chunk_respects_size_overlaps_and_never_splits_words() -> None:
    words = [f"w{i:03d}" for i in range(400)]  # 400 unique 4-char words
    text = " ".join(words)
    chunks = chunk(text, size=100, overlap=30)

    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)
    vocab = set(words)
    for c in chunks:
        assert set(c.split()) <= vocab  # a cut word would not be in the vocabulary
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert nxt.split()[0] in prev.split()  # the next chunk starts inside the previous one
    assert chunks[0].split()[0] == words[0]
    assert chunks[-1].split()[-1] == words[-1]


def test_parse_feed_newest_first_including_summary_only_entry() -> None:
    entries = parse_feed((FIXTURES / "feed.xml").read_bytes())
    assert [e.url for e in entries] == [
        "https://latent.space/p/second-post",
        "https://latent.space/p/first-post",
    ]
    assert entries[0].title == "Second post"
    assert entries[0].published_at == datetime(2026, 11, 20, 9, 0, tzinfo=UTC)
    assert entries[1].author == "Jane Writer"


def test_extract_article_drops_page_chrome() -> None:
    html = (FIXTURES / "article.html").read_text()
    art = extract_article(html, "https://latent.space/p/first-post")
    assert art.title == "Why durable steps matter"
    assert "replays the function from the top" in art.text
    assert "Subscribe now" not in art.text
    assert len(art.content_hash) == 64


def test_vector_pack_roundtrip_is_float32_exact_and_small() -> None:
    from array import array

    from commonplace.sources.embed import pack, unpack

    v = [0.1 * i - 0.05 for i in range(1536)]
    assert unpack(pack(v)) == array("f", v).tolist()
    assert len(pack(v)) < 8300
