import json
import time
from unittest.mock import patch, MagicMock

import pytest
import requests

from crawler import (
    load_config,
    extract_links,
    extract_title,
    clean_text,
    crawl,
    fetch_page,
    DEFAULT_MAX_ARTICLES,
    DEFAULT_DELAY,
    DEFAULT_LINKS_PER_PAGE,
    DEFAULT_MIN_WORDS,
    MIN_DELAY,
)


# ---------------------------------------------------------------------------
# load_config tests
# ---------------------------------------------------------------------------

CONFIG_ALL_KEYS = {
    "max_articles": 250,
    "delay": 1.5,
    "links_per_page": 5,
    "min_words": 100,
}


def test_load_config_valid(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(CONFIG_ALL_KEYS), encoding="utf-8")
    cfg = load_config(str(p))
    assert cfg == CONFIG_ALL_KEYS


def test_load_config_missing_keys(tmp_path):
    p = tmp_path / "config.json"
    p.write_text('{"max_articles": 50}', encoding="utf-8")
    cfg = load_config(str(p))
    assert cfg["max_articles"] == 50
    assert cfg["delay"] == DEFAULT_DELAY
    assert cfg["links_per_page"] == DEFAULT_LINKS_PER_PAGE
    assert cfg["min_words"] == DEFAULT_MIN_WORDS


def test_load_config_delay_below_minimum(tmp_path):
    p = tmp_path / "config.json"
    p.write_text('{"delay": 0.1}', encoding="utf-8")
    cfg = load_config(str(p))
    assert cfg["delay"] == MIN_DELAY


def test_load_config_delay_at_minimum(tmp_path):
    p = tmp_path / "config.json"
    p.write_text('{"delay": 0.5}', encoding="utf-8")
    cfg = load_config(str(p))
    assert cfg["delay"] == 0.5


def test_load_config_file_not_found():
    cfg = load_config("/nonexistent/path/to/config.json")
    assert cfg["max_articles"] == DEFAULT_MAX_ARTICLES
    assert cfg["delay"] == DEFAULT_DELAY
    assert cfg["links_per_page"] == DEFAULT_LINKS_PER_PAGE


def test_load_config_invalid_json(tmp_path):
    p = tmp_path / "config.json"
    p.write_text("not valid json {{{", encoding="utf-8")
    cfg = load_config(str(p))
    assert cfg["max_articles"] == DEFAULT_MAX_ARTICLES
    assert cfg["delay"] == DEFAULT_DELAY
    assert cfg["links_per_page"] == DEFAULT_LINKS_PER_PAGE


def test_load_config_min_words(tmp_path):
    p = tmp_path / "config.json"
    p.write_text('{"min_words": 50}', encoding="utf-8")
    cfg = load_config(str(p))
    assert cfg["min_words"] == 50
    assert cfg["max_articles"] == DEFAULT_MAX_ARTICLES


def test_load_config_empty_file(tmp_path):
    p = tmp_path / "config.json"
    p.write_text("", encoding="utf-8")
    cfg = load_config(str(p))
    assert cfg["max_articles"] == DEFAULT_MAX_ARTICLES


# ---------------------------------------------------------------------------
# extract_links tests
# ---------------------------------------------------------------------------

BASE_URL = "https://pt.wikipedia.org"


def test_extract_links_wiki_paths():
    html = '<a href="/wiki/Artigo1">A1</a><a href="/wiki/Artigo2">A2</a>'
    links = extract_links(html, BASE_URL)
    assert len(links) == 2
    assert "https://pt.wikipedia.org/wiki/Artigo1" in links
    assert "https://pt.wikipedia.org/wiki/Artigo2" in links


def test_extract_links_excludes_colon():
    html = (
        '<a href="/wiki/Artigo">OK</a>'
        '<a href="/wiki/Especial:Pesquisar">Bad</a>'
        '<a href="/wiki/Wikipedia:Sobre">Bad</a>'
    )
    links = extract_links(html, BASE_URL)
    assert len(links) == 1
    assert "Artigo" in links[0]


def test_extract_links_excludes_main_page():
    html = (
        '<a href="/wiki/Main_Page">MP</a>'
        '<a href="/wiki/Main_Page/">MP/</a>'
        '<a href="/wiki/Real_Article">Real</a>'
    )
    links = extract_links(html, BASE_URL)
    assert len(links) == 1
    assert "Real_Article" in links[0]


def test_extract_links_protocol_relative():
    html = '<a href="//pt.wikipedia.org/wiki/Artigo">A</a>'
    links = extract_links(html, BASE_URL)
    assert len(links) == 1
    assert links[0] == "https://pt.wikipedia.org/wiki/Artigo"


def test_extract_links_respects_limit():
    tags = "".join(f'<a href="/wiki/Art{i}">A{i}</a>' for i in range(15))
    html = f"<body>{tags}</body>"
    links = extract_links(html, BASE_URL, links_per_page=5)
    assert len(links) == 5


def test_extract_links_no_duplicates():
    html = (
        '<a href="/wiki/Artigo">A1</a>'
        '<a href="/wiki/Artigo">A2</a>'
        '<a href="/wiki/Artigo">A3</a>'
    )
    links = extract_links(html, BASE_URL)
    assert len(links) == 1


def test_extract_links_excludes_non_wiki():
    html = (
        '<a href="/outra-coisa">Nope</a>'
        '<a href="https://google.com">Nope</a>'
        '<a href="/wiki/Artigo">Yes</a>'
    )
    links = extract_links(html, BASE_URL)
    assert len(links) == 1
    assert "Artigo" in links[0]


def test_extract_links_other_domain():
    html = '<a href="/wiki/Artigo">A</a>'
    links = extract_links(html, "https://en.wikipedia.org")
    assert links[0] == "https://en.wikipedia.org/wiki/Artigo"


def test_extract_links_excludes_different_host():
    html = (
        '<a href="/wiki/Artigo">OK</a>'
        '<a href="//af.wikipedia.org/wiki/Outro">Bad</a>'
        '<a href="https://ar.wikipedia.org/wiki/Outro">Bad</a>'
    )
    links = extract_links(html, BASE_URL)
    assert len(links) == 1
    assert "pt.wikipedia.org" in links[0]


# ---------------------------------------------------------------------------
# extract_title tests
# ---------------------------------------------------------------------------


def test_extract_title_from_h1():
    html = "<html><body><h1>Titulo do Artigo</h1></body></html>"
    assert extract_title(html) == "Titulo do Artigo"


def test_extract_title_fallback_to_title_tag():
    html = "<html><head><title>Artigo - Wikipedia</title></head><body></body></html>"
    assert extract_title(html) == "Artigo - Wikipedia"


def test_extract_title_unknown():
    html = "<html><body>No title here</body></html>"
    assert extract_title(html) == "Unknown Title"


# ---------------------------------------------------------------------------
# clean_text tests
# ---------------------------------------------------------------------------


def test_clean_text_removes_scripts_styles():
    html = (
        "<html><body><div class='mw-parser-output'>"
        "<script>alert(1)</script>"
        "<style>.x{}</style>"
        "<noscript>no js</noscript>"
        "<p>Visible text</p>"
        "</div></body></html>"
    )
    text = clean_text(html)
    assert "Visible text" in text
    assert "alert(1)" not in text
    assert ".x{}" not in text
    assert "no js" not in text


def test_clean_text_removes_reference_classes():
    html = (
        "<html><body><div class='mw-parser-output'>"
        "<p>Keep this</p>"
        "<div class='reflist'>refs</div>"
        "<span class='reference'>cit</span>"
        "<div class='navbox'>nav</div>"
        "<div class='infobox'>info</div>"
        "<div class='metadata'>meta</div>"
        "</div></body></html>"
    )
    text = clean_text(html)
    assert "Keep this" in text
    assert "refs" not in text
    assert "cit" not in text
    assert "nav" not in text
    assert "info" not in text
    assert "meta" not in text


def test_clean_text_uses_mw_parser_output():
    html = (
        "<html><body>"
        "<div class='mw-parser-output'><p>Content</p></div>"
        "<div id='bodyContent'><p>Body content</p></div>"
        "</body></html>"
    )
    text = clean_text(html)
    assert "Content" in text
    assert "Body content" not in text


def test_clean_text_fallback_to_body_content():
    html = (
        "<html><body>"
        "<div id='bodyContent'><p>Fallback content</p></div>"
        "</body></html>"
    )
    text = clean_text(html)
    assert "Fallback content" in text


def test_clean_text_fallback_to_body():
    html = "<html><body><p>Body level</p></body></html>"
    text = clean_text(html)
    assert "Body level" in text


# ---------------------------------------------------------------------------
# crawl integration tests
# ---------------------------------------------------------------------------

PAGE_WITH_LINKS = """
<html>
<body>
<div class="mw-parser-output">
<h1>Test Page</h1>
<p>Content</p>
<a href="/wiki/Link1">L1</a>
<a href="/wiki/Link2">L2</a>
</div>
</body>
</html>
"""

PAGE_NO_LINKS = """
<html>
<body>
<div class="mw-parser-output">
<h1>Leaf Page</h1>
<p>No links here.</p>
</div>
</body>
</html>
"""


def test_crawl_respects_max_articles(tmp_path):
    output = tmp_path / "out.txt"
    with patch.object(requests.Session, "get") as mock_get, patch("crawler.time.sleep"):
        mock_get.return_value.text = PAGE_WITH_LINKS
        mock_get.return_value.raise_for_status = MagicMock()

        count, errors = crawl("https://pt.wikipedia.org/wiki/Start", max_articles=3, output_path=str(output), delay=0, links_per_page=2, min_words=1)
        assert count == 3
        assert errors == 0


def test_crawl_handles_http_error(tmp_path):
    output = tmp_path / "out.txt"
    with patch.object(requests.Session, "get") as mock_get, patch("crawler.time.sleep"):
        mock_get.side_effect = requests.RequestException("fail")

        count, errors = crawl("https://pt.wikipedia.org/wiki/Start", max_articles=2, output_path=str(output), delay=0, min_words=1)
        assert count == 0
        assert errors == 11


def test_crawl_skips_duplicates(tmp_path):
    output = tmp_path / "out.txt"
    call_count = {"count": 0}

    def fake_get(url, **kwargs):
        call_count["count"] += 1
        m = MagicMock()
        if call_count["count"] == 1:
            m.text = '<a href="/wiki/Link1">L1</a>'
        else:
            m.text = '<p>ok</p>'
        m.raise_for_status = MagicMock()
        return m

    with patch.object(requests.Session, "get", side_effect=fake_get), patch("crawler.time.sleep"):
        count, errors = crawl("https://pt.wikipedia.org/wiki/Start", max_articles=5, output_path=str(output), delay=0, links_per_page=5, min_words=1)
        assert count == 5


def test_crawl_uses_delay(tmp_path):
    output = tmp_path / "out.txt"
    delay_value = 2.0

    with patch.object(requests.Session, "get") as mock_get, patch("crawler.time.sleep") as mock_sleep:
        mock_get.return_value.text = PAGE_NO_LINKS
        mock_get.return_value.raise_for_status = MagicMock()

        crawl("https://pt.wikipedia.org/wiki/Start", max_articles=1, output_path=str(output), delay=delay_value, min_words=1)
        mock_sleep.assert_called_with(delay_value)


def test_crawl_stops_when_queue_empty(tmp_path):
    output = tmp_path / "out.txt"
    with patch.object(requests.Session, "get") as mock_get, patch("crawler.time.sleep"):
        mock_get.return_value.text = PAGE_NO_LINKS
        mock_get.return_value.raise_for_status = MagicMock()

        count, errors = crawl("https://pt.wikipedia.org/wiki/Start", max_articles=10, output_path=str(output), delay=0, min_words=1)
        assert count == 10


def test_crawl_limits_links_per_page(tmp_path):
    output = tmp_path / "out.txt"
    counter = {"n": 0}

    def fake_get(url, **kwargs):
        m = MagicMock()
        counter["n"] += 1
        base = counter["n"] * 10
        links = "".join(f'<a href="/wiki/Page{base + i}">A</a>' for i in range(10))
        m.text = links
        m.raise_for_status = MagicMock()
        return m

    with patch.object(requests.Session, "get", side_effect=fake_get), patch("crawler.time.sleep"):
        count, errors = crawl("https://pt.wikipedia.org/wiki/Start", max_articles=5, output_path=str(output), delay=0, links_per_page=3, min_words=1)
        assert count == 5


# ---------------------------------------------------------------------------
# output file tests
# ---------------------------------------------------------------------------


def test_crawl_writes_output_file(tmp_path):
    output = tmp_path / "out.txt"
    with patch.object(requests.Session, "get") as mock_get, patch("crawler.time.sleep"):
        mock_get.return_value.text = PAGE_NO_LINKS
        mock_get.return_value.raise_for_status = MagicMock()

        crawl("https://pt.wikipedia.org/wiki/Start", max_articles=1, output_path=str(output), delay=0, min_words=1)
        content = output.read_text(encoding="utf-8")

    assert "===== ARTICLE START =====" in content
    assert "===== ARTICLE END =====" in content


def test_crawl_output_contains_title_and_url(tmp_path):
    output = tmp_path / "out.txt"
    with patch.object(requests.Session, "get") as mock_get, patch("crawler.time.sleep"):
        mock_get.return_value.text = PAGE_NO_LINKS
        mock_get.return_value.raise_for_status = MagicMock()

        crawl("https://pt.wikipedia.org/wiki/Start", max_articles=1, output_path=str(output), delay=0, min_words=1)
        content = output.read_text(encoding="utf-8")

    assert "Title: Leaf Page" in content
    assert "URL: https://pt.wikipedia.org/wiki/Start" in content


def test_crawl_error_increments_counter(tmp_path):
    output = tmp_path / "out.txt"
    failing_urls = set()

    def fake_get(url, **kwargs):
        m = MagicMock()
        if url.rstrip("/") == "https://pt.wikipedia.org/wiki/Link1":
            failing_urls.add(url)
            raise requests.RequestException("fail")
        m.text = PAGE_WITH_LINKS
        m.raise_for_status = MagicMock()
        return m

    with patch.object(requests.Session, "get", side_effect=fake_get), patch("crawler.time.sleep"):
        count, errors = crawl("https://pt.wikipedia.org/wiki/Start", max_articles=5, output_path=str(output), delay=0, links_per_page=1, min_words=1)
        assert count == 5
        assert errors == 1


# ---------------------------------------------------------------------------
# fetch_page retry tests
# ---------------------------------------------------------------------------


def test_fetch_page_retries_on_timeout():
    session = MagicMock()
    session.get.side_effect = [
        requests.Timeout("timeout"),
        requests.Timeout("timeout"),
        MagicMock(text="<html>ok</html>", raise_for_status=MagicMock()),
    ]
    with patch("crawler.time.sleep"):
        result = fetch_page("https://pt.wikipedia.org/wiki/Test", session, max_retries=2)
    assert result == "<html>ok</html>"
    assert session.get.call_count == 3


def test_fetch_page_fails_after_max_retries():
    session = MagicMock()
    session.get.side_effect = requests.Timeout("timeout")
    with patch("crawler.time.sleep"):
        with pytest.raises(requests.Timeout):
            fetch_page("https://pt.wikipedia.org/wiki/Test", session, max_retries=2)
    assert session.get.call_count == 3


def test_fetch_page_does_not_retry_on_404():
    session = MagicMock()
    response_404 = MagicMock()
    response_404.raise_for_status.side_effect = requests.HTTPError(response=MagicMock(status_code=404))
    session.get.return_value = response_404
    with patch("crawler.time.sleep"):
        with pytest.raises(requests.HTTPError):
            fetch_page("https://pt.wikipedia.org/wiki/Test", session, max_retries=2)
    assert session.get.call_count == 1


def test_fetch_page_retries_on_500():
    session = MagicMock()
    response_500 = MagicMock()
    response_500.raise_for_status.side_effect = requests.HTTPError(response=MagicMock(status_code=500))
    response_ok = MagicMock(text="<html>ok</html>", raise_for_status=MagicMock())
    session.get.side_effect = [response_500, response_500, response_ok]
    with patch("crawler.time.sleep"):
        result = fetch_page("https://pt.wikipedia.org/wiki/Test", session, max_retries=2)
    assert result == "<html>ok</html>"
    assert session.get.call_count == 3


def test_fetch_page_retries_on_connection_error():
    session = MagicMock()
    session.get.side_effect = [
        requests.ConnectionError("conn refused"),
        MagicMock(text="<html>ok</html>", raise_for_status=MagicMock()),
    ]
    with patch("crawler.time.sleep"):
        result = fetch_page("https://pt.wikipedia.org/wiki/Test", session, max_retries=1)
    assert result == "<html>ok</html>"
    assert session.get.call_count == 2


# ---------------------------------------------------------------------------
# crawl word count tests
# ---------------------------------------------------------------------------

LONG_PAGE = """
<html>
<body>
<div class="mw-parser-output">
<h1>Long Article</h1>
<p>""" + " ".join(["word"] * 250) + """</p>
<a href="/wiki/Link1">L1</a>
<a href="/wiki/Link2">L2</a>
</div>
</body>
</html>
"""

SHORT_PAGE = """
<html>
<body>
<div class="mw-parser-output">
<h1>Short Article</h1>
<p>Too short</p>
<a href="/wiki/Link1">L1</a>
</div>
</body>
</html>
"""


def test_crawl_skips_short_article(tmp_path):
    output = tmp_path / "out.txt"
    with patch.object(requests.Session, "get") as mock_get, patch("crawler.time.sleep"):
        mock_response = MagicMock()
        mock_response.text = SHORT_PAGE
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        count, errors = crawl("https://pt.wikipedia.org/wiki/Start", max_articles=5, output_path=str(output), delay=0, min_words=200)
        assert count == 0


def test_crawl_reaches_exact_count_with_short_articles(tmp_path):
    output = tmp_path / "out.txt"
    pages = [SHORT_PAGE, LONG_PAGE, SHORT_PAGE, LONG_PAGE, SHORT_PAGE, LONG_PAGE, LONG_PAGE]
    call_count = {"count": 0}

    def fake_get(url, **kwargs):
        idx = min(call_count["count"], len(pages) - 1)
        call_count["count"] += 1
        m = MagicMock()
        m.text = pages[idx]
        m.raise_for_status = MagicMock()
        return m

    with patch.object(requests.Session, "get", side_effect=fake_get), patch("crawler.time.sleep"):
        count, errors = crawl("https://pt.wikipedia.org/wiki/Start", max_articles=3, output_path=str(output), delay=0, min_words=200)
        assert count == 3
        assert errors == 0


def test_crawl_fallback_when_queue_empty(tmp_path):
    output = tmp_path / "out.txt"
    requested_urls = []
    long_page = LONG_PAGE
    no_links_page = LONG_PAGE.replace('<a href="/wiki/Link1">L1</a>', "").replace('<a href="/wiki/Link2">L2</a>', "")

    def fake_get(url, **kwargs):
        requested_urls.append(url)
        m = MagicMock()
        if len(requested_urls) == 1:
            m.text = no_links_page
        else:
            m.text = long_page
        m.raise_for_status = MagicMock()
        return m

    with patch.object(requests.Session, "get", side_effect=fake_get), patch("crawler.time.sleep"):
        count, errors = crawl("https://pt.wikipedia.org/wiki/Start", max_articles=3, output_path=str(output), delay=0, min_words=200)
        assert count == 3
        assert "Special:Random" in str(requested_urls)


def test_crawl_exact_count_guaranteed(tmp_path):
    output = tmp_path / "out.txt"

    def fake_get(url, **kwargs):
        m = MagicMock()
        m.text = LONG_PAGE
        m.raise_for_status = MagicMock()
        return m

    with patch.object(requests.Session, "get", side_effect=fake_get), patch("crawler.time.sleep"):
        count, errors = crawl("https://pt.wikipedia.org/wiki/Start", max_articles=10, output_path=str(output), delay=0, links_per_page=3, min_words=200)
        assert count == 10
        assert errors == 0
