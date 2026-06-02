import sys
import json
import time
import logging
import argparse
from collections import deque
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

USER_AGENT = "Mozilla/5.0 (compatible; WikiCrawler/1.0; educational purposes)"
OUTPUT_FILE = "artigos_wikipedia.txt"
LOG_FILE = "crawler.log"
DEFAULT_CONFIG_PATH = "config.json"

DEFAULT_MAX_ARTICLES = 1000
DEFAULT_DELAY = 0.5
DEFAULT_LINKS_PER_PAGE = 10
DEFAULT_MIN_WORDS = 200
MIN_DELAY = 0.5
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2

logger = logging.getLogger("wikicrawler")


def _default_config():
    return {
        "max_articles": DEFAULT_MAX_ARTICLES,
        "delay": DEFAULT_DELAY,
        "links_per_page": DEFAULT_LINKS_PER_PAGE,
        "min_words": DEFAULT_MIN_WORDS,
    }


def setup_logging(log_path, verbose):
    logger.setLevel(logging.DEBUG)

    console_level = logging.DEBUG if verbose else logging.INFO
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)-7s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(file_handler)


def load_config(path):
    config = _default_config()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logger.warning("Could not read config from '%s', using defaults", path)
        return config

    config["max_articles"] = int(data.get("max_articles", config["max_articles"]))
    config["delay"] = float(data.get("delay", config["delay"]))
    config["links_per_page"] = int(data.get("links_per_page", config["links_per_page"]))
    config["min_words"] = int(data.get("min_words", config["min_words"]))

    if config["delay"] < MIN_DELAY:
        logger.warning(
            "delay=%s is below minimum %s, using %s", config["delay"], MIN_DELAY, MIN_DELAY
        )
        config["delay"] = MIN_DELAY

    return config


def extract_links(html, base_url, links_per_page=DEFAULT_LINKS_PER_PAGE):
    base_host = urlparse(base_url).hostname
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for tag in soup.find_all("a", href=True):
        href = tag["href"]
        full_url = urljoin(base_url, href)
        parsed = urlparse(full_url)
        if not parsed.path.startswith("/wiki/"):
            continue
        if parsed.hostname != base_host:
            continue
        if ":" in parsed.path:
            continue
        if parsed.path in ("/wiki/Main_Page", "/wiki/Main_Page/"):
            continue
        canonical = parsed.scheme + "://" + parsed.hostname + parsed.path
        if canonical not in links:
            links.append(canonical)
        if len(links) >= links_per_page:
            break
    return links


def fetch_page(url, session, max_retries=MAX_RETRIES):
    headers = {"User-Agent": USER_AGENT}
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            resp = session.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            return resp.text
        except requests.Timeout as e:
            last_error = e
            logger.warning("Timeout fetching %s (attempt %d/%d)", url, attempt + 1, max_retries + 1)
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status in (404, 410):
                raise
            last_error = e
            logger.warning("HTTP %s fetching %s (attempt %d/%d)", status, url, attempt + 1, max_retries + 1)
        except requests.ConnectionError as e:
            last_error = e
            logger.warning("Connection error fetching %s (attempt %d/%d)", url, attempt + 1, max_retries + 1)
        except requests.RequestException as e:
            last_error = e
            logger.warning("Error fetching %s (attempt %d/%d): %s", url, attempt + 1, max_retries + 1, e)

        if attempt < max_retries:
            wait = RETRY_BACKOFF_BASE ** (attempt + 1)
            logger.info("Retrying in %ds...", wait)
            time.sleep(wait)

    raise last_error


def crawl(start_url, max_articles, output_path, delay=DEFAULT_DELAY, links_per_page=DEFAULT_LINKS_PER_PAGE, min_words=DEFAULT_MIN_WORDS):
    session = requests.Session()
    queue = deque([start_url])
    visited = set()

    logger.info("Starting crawl from: %s", start_url)
    logger.info("Max articles: %s", max_articles)
    logger.info("Min words: %s", min_words)
    logger.info("Delay: %ss", delay)
    logger.info("Links per page: %s", links_per_page)

    count = 0
    errors = 0
    fallback_attempts = 0
    max_fallback_attempts = 10
    with open(output_path, "w", encoding="utf-8") as out:
        while count < max_articles:
            if not queue:
                fallback_attempts += 1
                if fallback_attempts > max_fallback_attempts:
                    logger.warning("Max fallback attempts reached. Stopping at %d/%d articles.", count, max_articles)
                    break
                fallback_url = f"https://pt.wikipedia.org/wiki/Special:Random?seed={fallback_attempts}"
                logger.warning("Queue empty with %d/%d articles. Adding fallback seed (attempt %d).", count, max_articles, fallback_attempts)
                queue.append(fallback_url)

            url = queue.popleft()
            if url in visited:
                continue

            logger.info("[%s/%s] Fetching: %s", count + 1, max_articles, url)
            try:
                html = fetch_page(url, session)
            except requests.HTTPError as e:
                status = e.response.status_code if e.response is not None else "?"
                if status in (404, 410):
                    logger.warning("Skipping %s (HTTP %s)", url, status)
                else:
                    logger.error("HTTP error fetching %s: %s", url, e)
                visited.add(url)
                errors += 1
                continue
            except requests.RequestException as e:
                logger.error("HTTP error fetching %s: %s", url, e)
                visited.add(url)
                errors += 1
                continue

            visited.add(url)

            title = extract_title(html)
            text = clean_text(html)

            word_count = len(text.split())
            if word_count < min_words:
                logger.info("Skipping short article (%d words < %d): %s", word_count, min_words, url)
                try:
                    links = extract_links(html, url, links_per_page)
                except Exception as e:
                    logger.error("Error extracting links from %s: %s", url, e)
                    links = []
                for link in links:
                    if link not in visited:
                        queue.append(link)
                time.sleep(delay)
                continue

            count += 1

            out.write("===== ARTICLE START =====\n")
            out.write(f"Title: {title}\n")
            out.write(f"URL: {url}\n")
            out.write("=========================\n")
            out.write(text)
            out.write("\n===== ARTICLE END =====\n\n")

            logger.info("Written article %s: %s (%s)", count, title, url)

            try:
                links = extract_links(html, url, links_per_page)
            except Exception as e:
                logger.error("Error extracting links from %s: %s", url, e)
                links = []

            for link in links:
                if link not in visited:
                    queue.append(link)

            time.sleep(delay)

    logger.info("Crawl finished. %s articles written, %s errors.", count, errors)

    if errors > 0:
        logger.warning("%s URLs failed during crawl.", errors)

    return count, errors


def extract_title(html):
    soup = BeautifulSoup(html, "html.parser")
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)
    title = soup.find("title")
    if title:
        return title.get_text(strip=True)
    return "Unknown Title"


def clean_text(html):
    soup = BeautifulSoup(html, "html.parser")

    content_div = soup.find("div", class_="mw-parser-output")
    if content_div is None:
        content_div = soup.find("div", id="bodyContent")
    if content_div is None:
        content_div = soup.find("body")
    if content_div is None:
        content_div = soup

    for tag in content_div.find_all(["script", "style", "noscript"]):
        tag.decompose()

    for tag in content_div.find_all(class_=["reflist", "reference", "navbox", "infobox", "metadata"]):
        tag.decompose()

    text = content_div.get_text(separator="\n", strip=True)

    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            lines.append(stripped)

    paragraphs = []
    current = []
    for line in lines:
        if len(line) < 80 and line.endswith((".", "!", "?", ":", ")")):
            current.append(line)
            if len(" ".join(current)) > 80:
                paragraphs.append(" ".join(current))
                current = []
        else:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            paragraphs.append(line)

    if current:
        paragraphs.append(" ".join(current))

    return "\n\n".join(paragraphs)


def main():
    parser = argparse.ArgumentParser(description="Wikipedia article crawler")
    parser.add_argument("url", help="Starting Wikipedia article URL")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to config.json")
    parser.add_argument("--log", default=LOG_FILE, help="Path to log file (default: crawler.log)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show debug-level messages on console")
    args = parser.parse_args()

    setup_logging(args.log, args.verbose)

    config = load_config(args.config)
    logger.info("Configuration loaded: %s", config)

    count, errors = crawl(
        args.url,
        max_articles=config["max_articles"],
        output_path=OUTPUT_FILE,
        delay=config["delay"],
        links_per_page=config["links_per_page"],
        min_words=config["min_words"],
    )
    if count == 0:
        logger.warning("No articles collected.")
        sys.exit(1)


if __name__ == "__main__":
    main()
