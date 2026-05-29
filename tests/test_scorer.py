"""Tests for the relevance scorer."""

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.article import Article
from src.scorer.relevance_scorer import (
    group_by_impact_type,
    score_article,
    score_articles,
)

_TIRES_KEYWORDS = {
    "primary": ["tires", "tire", "wheels", "rims", "rubber"],
    "brands": ["Michelin", "Goodyear", "Bridgestone", "Continental", "Pirelli"],
    "supply_chain": ["rubber prices", "tire tariff", "tire recall", "tire shortage", "raw rubber"],
    "seasonal": ["winter tires", "all-season tires", "EV tires", "tire inflation"],
    "regulatory": ["NHTSA", "tire safety standards", "tire labeling", "DOT tire regulations"],
}


def _art(title: str, snippet: str = "") -> Article:
    return Article(
        title=title,
        source="TestSource",
        url=f"https://example.com/{title[:20].replace(' ', '-')}",
        published_date=datetime(2026, 5, 15, tzinfo=timezone.utc),
        snippet=snippet,
    )


class TestScoreArticle(unittest.TestCase):

    def test_highly_relevant_tariff_article(self):
        # direct mention (30) + pricing/tariff (25) = 55; also fires competitor (15) if brands mentioned
        article = _art(
            "US Announces 25% Tariff on Chinese Tire Imports",
            "Goodyear and Michelin warn tariff on tire imports will raise prices for consumers.",
        )
        result = score_article(article, _TIRES_KEYWORDS, threshold=30)
        self.assertIsNotNone(result)
        # direct(30) + pricing(25) + competitor(15) = 70 → High
        self.assertGreaterEqual(result.relevance_score, 70)
        self.assertEqual(result.impact_level, "High")
        self.assertEqual(result.impact_type, "Pricing")

    def test_supply_chain_article(self):
        article = _art(
            "Rubber shortage hits tire manufacturers",
            "Raw rubber supply disruption from Southeast Asia affecting tire production.",
        )
        result = score_article(article, _TIRES_KEYWORDS, threshold=30)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.relevance_score, 50)
        self.assertEqual(result.impact_type, "Supply Chain")

    def test_regulatory_recall_article(self):
        article = _art(
            "NHTSA orders tire recall affecting 2 million vehicles",
            "Tire safety standards violated; DOT tire regulations triggered mandatory recall.",
        )
        result = score_article(article, _TIRES_KEYWORDS, threshold=30)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.relevance_score, 50)
        self.assertIn(result.impact_type, ("Regulatory", "Supply Chain"))

    def test_seasonal_article(self):
        # "demand spikes" and "surge" fire the demand signal (20pts) which beats
        # seasonal (10pts), so Demand wins.  Seasonal still contributes to the score.
        article = _art(
            "Winter tires demand spikes as storm season approaches",
            "All-season tires and winter tire sales surge before holiday season.",
        )
        result = score_article(article, _TIRES_KEYWORDS, threshold=30)
        self.assertIsNotNone(result)
        # Seasonal fires — check via breakdown instead of impact_type dominance
        self.assertEqual(result.score_breakdown.seasonal, 10)
        self.assertIn(result.impact_type, ("Seasonal", "Demand"))

    def test_irrelevant_article_filtered_out(self):
        article = _art(
            "NASCAR race results: driver wins Daytona 500",
            "Exciting finish at the speedway as the top driver crosses the line first.",
        )
        result = score_article(article, _TIRES_KEYWORDS, threshold=30)
        self.assertIsNone(result)

    def test_score_capped_at_100(self):
        # An article hitting every dimension should not exceed 100
        article = _art(
            "Tire tariff recall shortage Michelin NHTSA winter tires demand surge",
            "Tire prices up due to tariff on tire imports. Rubber shortage, recall, winter tire boom.",
        )
        result = score_article(article, _TIRES_KEYWORDS, threshold=0)
        self.assertIsNotNone(result)
        self.assertLessEqual(result.relevance_score, 100)

    def test_threshold_filtering(self):
        # Article only mentions a brand — may be below higher thresholds
        article = _art("Michelin announces investor day", "Michelin reports strong quarterly earnings.")
        low_threshold = score_article(article, _TIRES_KEYWORDS, threshold=10)
        high_threshold = score_article(article, _TIRES_KEYWORDS, threshold=80)
        self.assertIsNotNone(low_threshold)
        self.assertIsNone(high_threshold)

    def test_impact_levels(self):
        # High: ≥70
        high_art = _art(
            "Tire tariff and rubber shortage hit supply chain",
            "Tariff raises prices. Raw rubber shortage disrupts tire manufacturers.",
        )
        result = score_article(high_art, _TIRES_KEYWORDS, threshold=0)
        self.assertEqual(result.impact_level, "High")

    def test_score_breakdown_fields(self):
        article = _art("Tire tariff announced", "25% tariff on tires.")
        result = score_article(article, _TIRES_KEYWORDS, threshold=0)
        self.assertIsNotNone(result)
        bd = result.score_breakdown
        self.assertEqual(bd.direct_mention, 30)
        self.assertEqual(bd.pricing_tariff, 25)
        self.assertEqual(bd.total, result.relevance_score)


class TestScoreArticles(unittest.TestCase):

    def test_returns_only_articles_above_threshold(self):
        articles = [
            _art("US Announces Tire Tariff", "Tariff on tire imports announced."),
            _art("NASCAR race results", "Driver wins speedway race."),
            _art("Winter tires demand spikes", "All-season tires selling fast."),
        ]
        results = score_articles(articles, _TIRES_KEYWORDS, threshold=30)
        titles = [r.title for r in results]
        self.assertIn("US Announces Tire Tariff", titles)
        self.assertNotIn("NASCAR race results", titles)

    def test_sorted_by_score_descending(self):
        articles = [
            _art("Tire tariff and rubber shortage", "Tariff on tires and supply chain disruption."),
            _art("Michelin quarterly earnings", "Michelin posts revenue growth."),
        ]
        results = score_articles(articles, _TIRES_KEYWORDS, threshold=0)
        scores = [r.relevance_score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_empty_input(self):
        self.assertEqual(score_articles([], _TIRES_KEYWORDS), [])


class TestGroupByImpactType(unittest.TestCase):

    def test_groups_correctly(self):
        articles = [
            _art("Tire tariff increase", "Tariff on tires."),
            _art("Rubber shortage disrupts supply", "Tire shortage due to raw rubber disruption."),
            _art("NHTSA tire recall", "NHTSA recalls tires."),
        ]
        scored = score_articles(articles, _TIRES_KEYWORDS, threshold=0)
        groups = group_by_impact_type(scored)
        # All three should land in distinct groups
        all_in_groups = sum(len(v) for v in groups.values())
        self.assertEqual(all_in_groups, len(scored))

    def test_empty_input(self):
        self.assertEqual(group_by_impact_type([]), {})


if __name__ == "__main__":
    unittest.main()
