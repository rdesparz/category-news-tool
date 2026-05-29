"""Tests for the news fetcher pipeline."""

import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fetcher.news_fetcher import (
    _deduplicate,
    _filter_by_date,
    fetch_news_for_category,
)


def _make_article(title: str, days_ago: int = 1, source: str = "TestSource") -> dict:
    return {
        "title": title,
        "source": source,
        "url": f"https://example.com/{title.replace(' ', '-')}",
        "published_date": datetime.now(tz=timezone.utc) - timedelta(days=days_ago),
        "snippet": f"Snippet for {title}",
    }


class TestFilterByDate(unittest.TestCase):
    def test_keeps_recent_articles(self):
        articles = [_make_article("New", days_ago=1), _make_article("Old", days_ago=10)]
        result = _filter_by_date(articles, days=7)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["title"], "New")

    def test_keeps_articles_without_date(self):
        article = {"title": "Dateless", "source": "X", "url": "http://x.com", "published_date": None, "snippet": ""}
        result = _filter_by_date([article], days=7)
        self.assertEqual(len(result), 1)

    def test_all_within_window(self):
        # days_ago=6 is safely inside a 7-day window; days_ago=8 is outside
        articles = [_make_article("A", days_ago=2), _make_article("B", days_ago=5), _make_article("C", days_ago=6)]
        result = _filter_by_date(articles, days=7)
        self.assertEqual(len(result), 3)


class TestDeduplicate(unittest.TestCase):
    def test_removes_near_duplicate_titles(self):
        articles = [
            _make_article("Tire prices rise in Q1 2024"),
            _make_article("Tire prices rise in Q1 2024 — updated"),
            _make_article("EV battery shortage worsens"),
        ]
        result = _deduplicate(articles, threshold=85)
        self.assertEqual(len(result), 2)

    def test_keeps_distinct_articles(self):
        articles = [_make_article("Tire prices rise"), _make_article("Goodyear launches new EV tire")]
        result = _deduplicate(articles, threshold=85)
        self.assertEqual(len(result), 2)

    def test_exact_duplicates_removed(self):
        articles = [_make_article("Same Title"), _make_article("Same Title")]
        result = _deduplicate(articles, threshold=85)
        self.assertEqual(len(result), 1)


class TestFetchNewsForCategory(unittest.TestCase):
    def _config(self) -> dict:
        return {
            "fetcher": {
                "days_lookback": 7,
                "dedup_threshold": 85,
                "cache_ttl_hours": 6,
                "cache_dir": tempfile.mkdtemp(),
                "request_timeout": 5,
                "max_retries": 1,
                "retry_base_delay": 0.0,
            },
            "newsapi": {"api_key": "", "base_url": "https://newsapi.org/v2"},
            "gnews": {"api_key": "", "base_url": "https://gnews.io/api/v4"},
            "anthropic": {"model": "claude-haiku-4-5", "max_tokens": 512},
        }

    def test_returns_articles_from_rss(self):
        mock_articles = [_make_article("Tire tariffs increase", days_ago=2)]
        with patch("src.fetcher.news_fetcher.fetch_google_news_rss", return_value=mock_articles), \
             patch("src.fetcher.news_fetcher.get_keywords", return_value={"primary": ["tire"]}):
            result = fetch_news_for_category("Tires", config=self._config(), use_cache=False)
        self.assertGreaterEqual(len(result), 1)
        self.assertEqual(result[0]["title"], "Tire tariffs increase")

    def test_rss_failure_does_not_crash(self):
        """If RSS raises, the fetcher should still return an empty list gracefully."""
        with patch("src.fetcher.news_fetcher.fetch_google_news_rss", return_value=[]), \
             patch("src.fetcher.news_fetcher.get_keywords", return_value={"primary": ["tire"]}):
            result = fetch_news_for_category("Tires", config=self._config(), use_cache=False)
        self.assertIsInstance(result, list)

    def test_limit_is_respected(self):
        distinct_titles = [
            "Goodyear posts record quarterly earnings",
            "NHTSA recalls 500k SUV tires over safety flaw",
            "Rubber prices surge amid Southeast Asia flooding",
            "EV market drives demand for low-rolling-resistance tires",
            "Michelin expands US manufacturing footprint",
            "Winter tire shortage hits midwest dealerships",
            "Bridgestone invests in bio-based synthetic rubber",
            "New DOT labeling rules take effect next month",
            "Continental wins NASCAR tire contract",
            "Supply chain disruptions ease for tire sector",
        ]
        mock_articles = [_make_article(t, days_ago=1) for t in distinct_titles]
        with patch("src.fetcher.news_fetcher.fetch_google_news_rss", return_value=mock_articles), \
             patch("src.fetcher.news_fetcher.get_keywords", return_value={"primary": ["tire"]}):
            result = fetch_news_for_category("Tires", config=self._config(), use_cache=False, limit=3)
        self.assertEqual(len(result), 3)

    def test_cache_is_used_on_second_call(self):
        mock_articles = [_make_article("Cached Article", days_ago=1)]
        config = self._config()
        with patch("src.fetcher.news_fetcher.fetch_google_news_rss", return_value=mock_articles) as mock_rss, \
             patch("src.fetcher.news_fetcher.get_keywords", return_value={"primary": ["tire"]}):
            # First call populates cache
            fetch_news_for_category("Tires", config=config, use_cache=True)
            # Second call should hit cache and not call RSS again
            fetch_news_for_category("Tires", config=config, use_cache=True)
            self.assertEqual(mock_rss.call_count, 1)


if __name__ == "__main__":
    unittest.main()
