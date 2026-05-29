"""Google News RSS fetcher — no API key required."""

import logging
import time
from datetime import datetime, timezone
from urllib.parse import quote_plus

import feedparser
import requests

logger = logging.getLogger(__name__)


def _entry_to_article(entry: dict) -> dict | None:
    """Convert a feedparser entry to our Article schema."""
    title = entry.get("title", "").strip()
    url = entry.get("link", "").strip()
    if not title or not url:
        return None

    # feedparser gives published_parsed as a time.struct_time in UTC
    published = None
    if entry.get("published_parsed"):
        published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)

    snippet = ""
    if entry.get("summary"):
        snippet = entry["summary"]
    elif entry.get("description"):
        snippet = entry["description"]
    # Strip basic HTML tags from snippet
    import re
    snippet = re.sub(r"<[^>]+>", " ", snippet).strip()[:500]

    source = ""
    if entry.get("source"):
        source = entry["source"].get("title", "")
    if not source and "google.com" in url:
        source = "Google News"

    return {
        "title": title,
        "source": source or "RSS",
        "url": url,
        "published_date": published,
        "snippet": snippet,
    }


def fetch_google_news_rss(
    query: str,
    timeout: int = 10,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> list[dict]:
    """
    Fetch articles from Google News RSS for a given query string.

    Returns a list of Article dicts. Never raises — returns [] on failure.
    """
    encoded = quote_plus(query)
    url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, timeout=timeout, headers={"User-Agent": "category-news-tool/1.0"})
            resp.raise_for_status()
            feed = feedparser.parse(resp.text)
            articles = []
            for entry in feed.entries:
                article = _entry_to_article(entry)
                if article:
                    articles.append(article)
            logger.debug("RSS query %r returned %d entries", query, len(articles))
            return articles
        except Exception as exc:
            wait = base_delay * (2 ** (attempt - 1))
            if attempt < max_retries:
                logger.warning("RSS attempt %d/%d failed for %r: %s — retrying in %.1fs", attempt, max_retries, query, exc, wait)
                time.sleep(wait)
            else:
                logger.error("RSS fetch failed for %r after %d attempts: %s", query, max_retries, exc)
                return []
    return []
