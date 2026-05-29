"""Tests for the article summarizer."""

import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, call, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.article import Article, ScoreBreakdown, ScoredArticle
from src.summarizer.summarizer import (
    _build_batches,
    _summarize_batch,
    group_summarized_by_impact,
    summarize_articles,
)


def _scored(title: str, score: int = 60, impact_type: str = "Pricing", url: str = "") -> ScoredArticle:
    article = Article(
        title=title,
        source="Reuters",
        url=url or f"https://example.com/{title[:20].replace(' ', '-')}",
        published_date=datetime(2026, 5, 15, tzinfo=timezone.utc),
        snippet=f"Snippet for {title}",
    )
    bd = ScoreBreakdown(pricing_tariff=25, direct_mention=30)
    return ScoredArticle(
        article=article,
        relevance_score=score,
        impact_level="High" if score >= 70 else "Medium",
        impact_type=impact_type,
        score_breakdown=bd,
    )


def _make_mock_client(summaries: dict) -> MagicMock:
    """Build a mock Anthropic client that returns *summaries* as JSON."""
    block = MagicMock()
    block.type = "text"
    block.text = json.dumps(summaries)

    usage = MagicMock()
    usage.input_tokens = 100
    usage.cache_creation_input_tokens = 20
    usage.cache_read_input_tokens = 0
    usage.output_tokens = 80

    message = MagicMock()
    message.content = [block]
    message.usage = usage

    client = MagicMock()
    client.messages.create.return_value = message
    return client


class TestBuildBatches(unittest.TestCase):

    def test_single_batch_for_few_articles(self):
        items = [(i, _scored(f"Article {i}")) for i in range(3)]
        batches = _build_batches(items)
        self.assertEqual(len(batches), 1)
        self.assertEqual(len(batches[0]), 3)

    def test_splits_at_max_articles(self):
        items = [(i, _scored(f"Article {i}")) for i in range(7)]
        batches = _build_batches(items)
        # With max 5 per batch: batch 1 = 5, batch 2 = 2
        self.assertEqual(len(batches), 2)
        self.assertEqual(len(batches[0]), 5)
        self.assertEqual(len(batches[1]), 2)

    def test_empty_input(self):
        self.assertEqual(_build_batches([]), [])


class TestSummarizeBatch(unittest.TestCase):

    def test_returns_summaries_keyed_by_index(self):
        articles = [_scored("Tire tariff"), _scored("Rubber shortage")]
        summaries = {"0": "Tariff summary.", "1": "Shortage summary."}
        client = _make_mock_client(summaries)

        result = _summarize_batch(
            batch=articles,
            batch_start_index=0,
            client=client,
            model="claude-haiku-4-5",
            max_output_tokens=512,
            max_retries=1,
            base_delay=0.0,
        )
        self.assertEqual(result[0], "Tariff summary.")
        self.assertEqual(result[1], "Shortage summary.")

    def test_non_zero_batch_start_index(self):
        articles = [_scored("Article 3")]
        summaries = {"3": "Summary for article 3."}
        client = _make_mock_client(summaries)

        result = _summarize_batch(
            batch=articles,
            batch_start_index=3,
            client=client,
            model="claude-haiku-4-5",
            max_output_tokens=512,
            max_retries=1,
            base_delay=0.0,
        )
        self.assertEqual(result[3], "Summary for article 3.")

    def test_system_prompt_has_cache_control(self):
        articles = [_scored("Test article")]
        client = _make_mock_client({"0": "Summary."})
        _summarize_batch(
            batch=articles, batch_start_index=0, client=client,
            model="claude-haiku-4-5", max_output_tokens=512,
            max_retries=1, base_delay=0.0,
        )
        kwargs = client.messages.create.call_args.kwargs
        system = kwargs["system"]
        self.assertEqual(system[0]["cache_control"]["type"], "ephemeral")

    def test_graceful_fallback_on_invalid_json(self):
        block = MagicMock()
        block.type = "text"
        block.text = "This is not JSON"
        usage = MagicMock()
        usage.input_tokens = 10
        usage.cache_creation_input_tokens = 0
        usage.cache_read_input_tokens = 0
        usage.output_tokens = 5
        message = MagicMock()
        message.content = [block]
        message.usage = usage
        client = MagicMock()
        client.messages.create.return_value = message

        articles = [_scored("Article")]
        result = _summarize_batch(
            batch=articles, batch_start_index=0, client=client,
            model="claude-haiku-4-5", max_output_tokens=512,
            max_retries=1, base_delay=0.0,
        )
        # Should not raise; returns the raw text as fallback
        self.assertIn(0, result)


class TestSummarizeArticles(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_returns_summarized_articles(self):
        articles = [_scored("Tire tariff"), _scored("Rubber shortage")]
        client = _make_mock_client({"0": "Tariff summary.", "1": "Shortage summary."})

        results = summarize_articles(
            articles, "Tires", client=client,
            model="claude-haiku-4-5", cache_dir=self.tmpdir,
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].summary, "Tariff summary.")
        self.assertEqual(results[1].summary, "Shortage summary.")

    def test_cache_prevents_second_api_call(self):
        url = "https://example.com/unique-article"
        articles = [_scored("Tire tariff", url=url)]
        client = _make_mock_client({"0": "Cached summary."})

        # First call — populates cache
        summarize_articles(articles, "Tires", client=client, cache_dir=self.tmpdir)
        # Second call — should use cache, not call the API again
        summarize_articles(articles, "Tires", client=client, cache_dir=self.tmpdir)
        self.assertEqual(client.messages.create.call_count, 1)

    def test_rate_limit_error_retried(self):
        articles = [_scored("Tire tariff")]
        client = MagicMock()
        import anthropic as _anthropic
        # Fail once then succeed
        success_block = MagicMock()
        success_block.type = "text"
        success_block.text = json.dumps({"0": "Summary after retry."})
        success_usage = MagicMock()
        success_usage.input_tokens = 10
        success_usage.cache_creation_input_tokens = 0
        success_usage.cache_read_input_tokens = 0
        success_usage.output_tokens = 5
        success_msg = MagicMock()
        success_msg.content = [success_block]
        success_msg.usage = success_usage

        client.messages.create.side_effect = [
            _anthropic.RateLimitError("rate limit", response=MagicMock(status_code=429), body={}),
            success_msg,
        ]

        results = summarize_articles(
            articles, "Tires", client=client,
            max_retries=2, base_delay=0.0, cache_dir=self.tmpdir,
        )
        self.assertEqual(results[0].summary, "Summary after retry.")
        self.assertEqual(client.messages.create.call_count, 2)

    def test_api_failure_falls_back_to_template(self):
        articles = [_scored("Tire tariff")]
        client = MagicMock()
        import anthropic as _anthropic
        client.messages.create.side_effect = _anthropic.RateLimitError(
            "rate limit", response=MagicMock(status_code=429), body={}
        )

        results = summarize_articles(
            articles, "Tires", client=client,
            max_retries=1, base_delay=0.0, cache_dir=self.tmpdir,
        )
        self.assertEqual(len(results), 1)
        # Template fallback uses the impact-type framing — Pricing impact
        # gets the ASP/margin sentence appended.
        summary = results[0].summary.lower()
        self.assertIn("asp", summary)

    def test_no_client_uses_templates(self):
        articles = [_scored("Tire tariff")]
        results = summarize_articles(
            articles, "Tires", client=None, cache_dir=self.tmpdir,
        )
        self.assertEqual(len(results), 1)
        # Template summary should mention pricing-impact framing
        self.assertIn("asp", results[0].summary.lower())

    def test_batching_sends_multiple_calls_for_many_articles(self):
        # 7 articles should produce 2 batches (5 + 2)
        articles = [_scored(f"Article {i}") for i in range(7)]
        # Mock returns summaries for all 7 across 2 calls
        batch1 = {str(i): f"Summary {i}." for i in range(5)}
        batch2 = {str(i): f"Summary {i}." for i in range(5, 7)}

        blocks = []
        for summaries in [batch1, batch2]:
            block = MagicMock()
            block.type = "text"
            block.text = json.dumps(summaries)
            usage = MagicMock()
            usage.input_tokens = 100
            usage.cache_creation_input_tokens = 10
            usage.cache_read_input_tokens = 0
            usage.output_tokens = 50
            msg = MagicMock()
            msg.content = [block]
            msg.usage = usage
            blocks.append(msg)

        client = MagicMock()
        client.messages.create.side_effect = blocks

        results = summarize_articles(
            articles, "Tires", client=client,
            cache_dir=self.tmpdir,
        )
        self.assertEqual(len(results), 7)
        self.assertEqual(client.messages.create.call_count, 2)

    def test_empty_input(self):
        client = MagicMock()
        results = summarize_articles([], "Tires", client=client, cache_dir=self.tmpdir)
        self.assertEqual(results, [])
        client.messages.create.assert_not_called()


class TestGroupSummarizedByImpact(unittest.TestCase):

    def test_groups_by_impact_type(self):
        articles = [
            _scored("Tariff article", impact_type="Pricing"),
            _scored("Shortage article", impact_type="Supply Chain"),
            _scored("Another pricing", impact_type="Pricing"),
        ]
        # Wrap in SummarizedArticle
        from src.models.article import SummarizedArticle
        summarized = [SummarizedArticle(scored=a, summary="s") for a in articles]

        groups = group_summarized_by_impact(summarized)
        self.assertIn("Pricing", groups)
        self.assertIn("Supply Chain", groups)
        self.assertEqual(len(groups["Pricing"]), 2)
        self.assertEqual(len(groups["Supply Chain"]), 1)

    def test_groups_sorted_by_score_descending(self):
        from src.models.article import SummarizedArticle
        articles = [
            _scored("Low score", score=40, impact_type="Pricing"),
            _scored("High score", score=85, impact_type="Pricing"),
        ]
        summarized = [SummarizedArticle(scored=a, summary="s") for a in articles]
        groups = group_summarized_by_impact(summarized)
        pricing = groups["Pricing"]
        self.assertEqual(pricing[0].relevance_score, 85)
        self.assertEqual(pricing[1].relevance_score, 40)


if __name__ == "__main__":
    unittest.main()
