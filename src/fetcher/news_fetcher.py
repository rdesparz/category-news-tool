"""
Category News Fetcher — main entry point.

Usage:
    python -m src.fetcher.news_fetcher --category "Tires"
    python -m src.fetcher.news_fetcher --category "Wireless Headphones" --limit 20
"""

import argparse
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
import yaml

from src.fetcher.rss_parser import fetch_google_news_rss
from src.keywords.category_keywords import flatten_keywords, get_keywords

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def setup_logging(config: dict) -> None:
    log_cfg = config.get("logging", {})
    logging.basicConfig(
        level=getattr(logging, log_cfg.get("level", "INFO")),
        format=log_cfg.get("format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s"),
    )


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _cache_key(category: str) -> str:
    return hashlib.md5(category.lower().encode()).hexdigest()


def _load_cache(category: str, cache_dir: str, ttl_hours: int) -> list[dict] | None:
    path = Path(cache_dir) / f"{_cache_key(category)}.json"
    if not path.exists():
        return None
    with open(path) as f:
        payload = json.load(f)
    saved_at = datetime.fromisoformat(payload["saved_at"])
    if datetime.now(tz=timezone.utc) - saved_at > timedelta(hours=ttl_hours):
        logger.debug("Cache expired for %r", category)
        return None
    logger.info("Cache hit for %r (%d articles)", category, len(payload["articles"]))
    # Re-parse date strings back to datetime objects
    for a in payload["articles"]:
        if a.get("published_date"):
            a["published_date"] = datetime.fromisoformat(a["published_date"])
    return payload["articles"]


def _save_cache(category: str, articles: list[dict], cache_dir: str) -> None:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    path = Path(cache_dir) / f"{_cache_key(category)}.json"
    serializable = []
    for a in articles:
        copy = dict(a)
        if isinstance(copy.get("published_date"), datetime):
            copy["published_date"] = copy["published_date"].isoformat()
        serializable.append(copy)
    with open(path, "w") as f:
        json.dump({"saved_at": datetime.now(tz=timezone.utc).isoformat(), "articles": serializable}, f)
    logger.debug("Cached %d articles for %r", len(articles), category)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _deduplicate(articles: list[dict], threshold: int) -> list[dict]:
    """Remove articles whose title is >threshold% similar to an already-seen title."""
    try:
        from thefuzz import fuzz
    except ImportError:
        logger.warning("thefuzz not installed — skipping deduplication")
        return articles

    kept: list[dict] = []
    seen_titles: list[str] = []
    for article in articles:
        title = article["title"]
        is_dup = any(fuzz.token_sort_ratio(title, seen) >= threshold for seen in seen_titles)
        if not is_dup:
            kept.append(article)
            seen_titles.append(title)
    removed = len(articles) - len(kept)
    if removed:
        logger.info("Deduplication removed %d duplicate articles", removed)
    return kept


# ---------------------------------------------------------------------------
# Date filtering
# ---------------------------------------------------------------------------

def _filter_by_date(articles: list[dict], days: int) -> list[dict]:
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
    kept = []
    for a in articles:
        pub = a.get("published_date")
        if pub is None:
            # Include articles with no date (we can't filter them out accurately)
            kept.append(a)
            continue
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        if pub >= cutoff:
            kept.append(a)
    removed = len(articles) - len(kept)
    if removed:
        logger.debug("Date filter removed %d articles older than %d days", removed, days)
    return kept


# ---------------------------------------------------------------------------
# NewsAPI source
# ---------------------------------------------------------------------------

def _fetch_newsapi(
    query: str,
    api_key: str,
    base_url: str,
    days: int,
    timeout: int,
    max_retries: int,
    base_delay: float,
) -> list[dict]:
    if not api_key:
        return []
    from_date = (datetime.now(tz=timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    params = {
        "q": query,
        "from": from_date,
        "sortBy": "publishedAt",
        "language": "en",
        "apiKey": api_key,
    }
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(f"{base_url}/everything", params=params, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            articles = []
            for item in data.get("articles", []):
                pub = None
                if item.get("publishedAt"):
                    try:
                        pub = datetime.fromisoformat(item["publishedAt"].replace("Z", "+00:00"))
                    except ValueError:
                        pass
                articles.append({
                    "title": item.get("title", "").strip(),
                    "source": item.get("source", {}).get("name", "NewsAPI"),
                    "url": item.get("url", ""),
                    "published_date": pub,
                    "snippet": (item.get("description") or "")[:500],
                })
            logger.debug("NewsAPI query %r returned %d articles", query, len(articles))
            return articles
        except Exception as exc:
            wait = base_delay * (2 ** (attempt - 1))
            if attempt < max_retries:
                logger.warning("NewsAPI attempt %d/%d failed for %r: %s — retrying in %.1fs", attempt, max_retries, query, exc, wait)
                time.sleep(wait)
            else:
                logger.error("NewsAPI failed for %r after %d attempts: %s", query, max_retries, exc)
                return []
    return []


# ---------------------------------------------------------------------------
# GNews source
# ---------------------------------------------------------------------------

def _fetch_gnews(
    query: str,
    api_key: str,
    base_url: str,
    days: int,
    timeout: int,
    max_retries: int,
    base_delay: float,
) -> list[dict]:
    if not api_key:
        return []
    params = {
        "q": query,
        "lang": "en",
        "country": "us",
        "max": 10,
        "token": api_key,
    }
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(f"{base_url}/search", params=params, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            articles = []
            for item in data.get("articles", []):
                pub = None
                if item.get("publishedAt"):
                    try:
                        pub = datetime.fromisoformat(item["publishedAt"].replace("Z", "+00:00"))
                    except ValueError:
                        pass
                articles.append({
                    "title": item.get("title", "").strip(),
                    "source": item.get("source", {}).get("name", "GNews"),
                    "url": item.get("url", ""),
                    "published_date": pub,
                    "snippet": (item.get("description") or "")[:500],
                })
            logger.debug("GNews query %r returned %d articles", query, len(articles))
            return articles
        except Exception as exc:
            wait = base_delay * (2 ** (attempt - 1))
            if attempt < max_retries:
                logger.warning("GNews attempt %d/%d failed for %r: %s — retrying in %.1fs", attempt, max_retries, query, exc, wait)
                time.sleep(wait)
            else:
                logger.error("GNews failed for %r after %d attempts: %s", query, max_retries, exc)
                return []
    return []


# ---------------------------------------------------------------------------
# Main fetch function
# ---------------------------------------------------------------------------

def fetch_news_for_category(
    category: str,
    config: dict | None = None,
    use_cache: bool = True,
    limit: int | None = None,
    days: int | None = None,
) -> list[dict]:
    """
    Fetch deduplicated, date-filtered articles for a product category.

    Returns list of Article dicts: {title, source, url, published_date, snippet}.
    If *days* is provided, it overrides config.fetcher.days_lookback.
    """
    if config is None:
        config = load_config()

    fetcher_cfg = config.get("fetcher", {})
    if days is None:
        days = fetcher_cfg.get("days_lookback", 7)
    dedup_threshold = fetcher_cfg.get("dedup_threshold", 85)
    cache_ttl = fetcher_cfg.get("cache_ttl_hours", 6)
    cache_dir = fetcher_cfg.get("cache_dir", ".cache")
    timeout = fetcher_cfg.get("request_timeout", 10)
    max_retries = fetcher_cfg.get("max_retries", 3)
    base_delay = fetcher_cfg.get("retry_base_delay", 1.0)

    newsapi_key = os.environ.get("NEWS_API_KEY", "") or config.get("newsapi", {}).get("api_key", "")
    gnews_key = os.environ.get("GNEWS_API_KEY", "") or config.get("gnews", {}).get("api_key", "")

    # Check cache first — keyed by (category, days) so different lookbacks don't collide
    cache_key_str = f"{category}__{days}d"
    if use_cache:
        cached = _load_cache(cache_key_str, cache_dir, cache_ttl)
        if cached is not None:
            return cached[:limit] if limit else cached

    # Resolve keywords
    keywords = get_keywords(category, config=config)
    all_terms = flatten_keywords(keywords)

    # Build multiple search queries to broaden coverage:
    #   1. Primary product terms
    #   2. Top brand names
    #   3. Supply chain / regulatory terms (specific events tied to the category)
    queries = _build_search_queries(keywords)
    logger.info("Fetching news for %r with %d queries: %s", category, len(queries), queries)

    articles: list[dict] = []

    for query in queries:
        # Source 1: Google News RSS
        articles.extend(
            fetch_google_news_rss(query, timeout=timeout, max_retries=max_retries, base_delay=base_delay)
        )

        # Source 2: NewsAPI
        articles.extend(
            _fetch_newsapi(
                query,
                api_key=newsapi_key,
                base_url=config.get("newsapi", {}).get("base_url", "https://newsapi.org/v2"),
                days=days,
                timeout=timeout,
                max_retries=max_retries,
                base_delay=base_delay,
            )
        )

        # Source 3: GNews
        articles.extend(
            _fetch_gnews(
                query,
                api_key=gnews_key,
                base_url=config.get("gnews", {}).get("base_url", "https://gnews.io/api/v4"),
                days=days,
                timeout=timeout,
                max_retries=max_retries,
                base_delay=base_delay,
            )
        )

    logger.info("Collected %d raw articles from all sources", len(articles))

    # Filter to last N days
    articles = _filter_by_date(articles, days)

    # Deduplicate
    articles = _deduplicate(articles, dedup_threshold)

    # Sort newest-first (articles without dates go last)
    articles.sort(
        key=lambda a: a.get("published_date") or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )

    logger.info("Returning %d articles for %r after filtering and deduplication", len(articles), category)

    if use_cache:
        _save_cache(cache_key_str, articles, cache_dir)

    return articles[:limit] if limit else articles


def _build_search_queries(keywords: dict) -> list[str]:
    """
    Build multiple targeted search queries to broaden coverage.

    Each query becomes one round of fetches across all sources. Splitting into
    separate queries — instead of one giant OR — gives better RSS results,
    especially for broad categories like 'Automotive' where a generic primary
    term returns mostly irrelevant news.
    """
    queries: list[str] = []

    primary = keywords.get("primary", [])
    brands = keywords.get("brands", [])
    supply = keywords.get("supply_chain", [])
    regulatory = keywords.get("regulatory", [])

    # Query 1: top primary product terms
    if primary:
        queries.append(" OR ".join(f'"{t}"' for t in primary[:3]))

    # Query 2: top brand names (high-signal news; brand earnings, recalls, launches)
    if brands:
        queries.append(" OR ".join(f'"{t}"' for t in brands[:5]))

    # Query 3: supply chain / regulatory terms (specific events tied to the category)
    event_terms = (supply + regulatory)[:5]
    if event_terms:
        queries.append(" OR ".join(f'"{t}"' for t in event_terms))

    if not queries:
        # Defensive fallback: nothing in any group
        all_terms = []
        for terms in keywords.values():
            all_terms.extend(terms)
        if all_terms:
            queries.append(" OR ".join(f'"{t}"' for t in all_terms[:3]))

    return queries


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _format_article(article: dict, index: int) -> str:
    pub = article.get("published_date")
    pub_str = pub.strftime("%Y-%m-%d") if pub else "unknown date"
    lines = [
        f"[{index}] {article['title']}",
        f"    Source : {article['source']}",
        f"    Date   : {pub_str}",
        f"    URL    : {article['url']}",
    ]
    if article.get("snippet"):
        lines.append(f"    Snippet: {article['snippet'][:120]}...")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch category news articles")
    parser.add_argument("--category", required=True, help="Product category name, e.g. 'Tires'")
    parser.add_argument("--limit", type=int, default=None, help="Max articles to return")
    parser.add_argument("--no-cache", action="store_true", help="Bypass cache")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)

    articles = fetch_news_for_category(
        args.category,
        config=config,
        use_cache=not args.no_cache,
        limit=args.limit,
    )

    if not articles:
        print(f"No articles found for category: {args.category!r}")
        return

    print(f"\n=== {len(articles)} articles for '{args.category}' (last 7 days) ===\n")
    for i, article in enumerate(articles, 1):
        print(_format_article(article, i))
        print()


if __name__ == "__main__":
    main()
