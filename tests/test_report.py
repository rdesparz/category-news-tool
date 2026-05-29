"""Tests for the report generator."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.article import Article, ScoreBreakdown, ScoredArticle, SummarizedArticle
from src.report.report_generator import (
    _render_html_inline,
    _render_json,
    _render_markdown,
    build_report_data,
    save_report,
)


def _summ(title: str, impact_type: str = "Pricing", level: str = "High", score: int = 80) -> SummarizedArticle:
    art = Article(
        title=title,
        source="Reuters",
        url=f"https://example.com/{title[:15].replace(' ', '-')}",
        published_date=datetime(2026, 5, 15, tzinfo=timezone.utc),
        snippet="Snippet.",
    )
    bd = ScoreBreakdown(pricing_tariff=25, direct_mention=30)
    scored = ScoredArticle(
        article=art,
        relevance_score=score,
        impact_level=level,
        impact_type=impact_type,
        score_breakdown=bd,
    )
    return SummarizedArticle(scored=scored, summary=f"Summary of {title}.")


def _mock_client(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    usage = MagicMock()
    usage.input_tokens = 50
    usage.cache_creation_input_tokens = 10
    usage.cache_read_input_tokens = 0
    usage.output_tokens = 30
    msg = MagicMock()
    msg.content = [block]
    msg.usage = usage
    client = MagicMock()
    client.messages.create.return_value = msg
    return client


class TestBuildReportData(unittest.TestCase):

    def test_structure_keys_present(self):
        articles = [_summ("Tariff news"), _summ("Shortage alert", "Supply Chain")]
        client = _mock_client("- Key insight one.\n- Key insight two.")
        data = build_report_data(
            category="Tires",
            articles_analyzed=20,
            summarized=articles,
            days=7,
            client=client,
            model="claude-haiku-4-5",
            max_retries=1,
            base_delay=0.0,
            sources_used=["Google News RSS"],
        )
        for key in ("category", "date_start", "date_end", "articles_analyzed",
                    "relevant_articles", "executive_summary", "sections",
                    "recommended_actions", "generated_at", "sources_used", "disclaimer"):
            self.assertIn(key, data)

    def test_articles_analyzed_count(self):
        articles = [_summ("Article")]
        client = _mock_client("- Insight.")
        data = build_report_data(
            category="Tires", articles_analyzed=50, summarized=articles,
            days=7, client=client, model="claude-haiku-4-5",
            max_retries=1, base_delay=0.0, sources_used=[],
        )
        self.assertEqual(data["articles_analyzed"], 50)
        self.assertEqual(data["relevant_articles"], 1)

    def test_no_articles_produces_fallback_messages(self):
        data = build_report_data(
            category="Tires", articles_analyzed=10, summarized=[],
            days=7, client=None, model="claude-haiku-4-5",
            max_retries=1, base_delay=0.0, sources_used=[],
        )
        self.assertIn("No relevant", data["executive_summary"][0])
        self.assertIn("No actionable", data["recommended_actions"][0])

    def test_sections_grouped_by_impact_type(self):
        articles = [
            _summ("Tariff A", "Pricing"),
            _summ("Tariff B", "Pricing"),
            _summ("Shortage", "Supply Chain"),
        ]
        client = _mock_client("- Insight.")
        data = build_report_data(
            category="Tires", articles_analyzed=20, summarized=articles,
            days=7, client=client, model="claude-haiku-4-5",
            max_retries=1, base_delay=0.0, sources_used=[],
        )
        types_in_report = [s["impact_type"] for s in data["sections"]]
        self.assertIn("Pricing", types_in_report)
        self.assertIn("Supply Chain", types_in_report)
        pricing_section = next(s for s in data["sections"] if s["impact_type"] == "Pricing")
        self.assertEqual(len(pricing_section["articles"]), 2)

    def test_llm_failure_produces_placeholder(self):
        import anthropic as _anth
        client = MagicMock()
        client.messages.create.side_effect = _anth.RateLimitError(
            "rl", response=MagicMock(status_code=429), body={}
        )
        articles = [_summ("Tariff article", "Pricing")]
        data = build_report_data(
            category="Tires", articles_analyzed=5, summarized=articles,
            days=7, client=client, model="claude-haiku-4-5",
            max_retries=1, base_delay=0.0, sources_used=[],
        )
        # On LLM failure we now fall back to template executive summary —
        # which surfaces the article title grouped by impact type.
        self.assertTrue(len(data["executive_summary"]) >= 1)
        self.assertIn("Tariff article", data["executive_summary"][0])
        # Recommended actions also fall back to templates
        self.assertTrue(len(data["recommended_actions"]) >= 1)


class TestRenderMarkdown(unittest.TestCase):

    def _data(self) -> dict:
        return {
            "category": "Tires",
            "date_start": "May 12, 2026",
            "date_end": "May 19, 2026",
            "articles_analyzed": 30,
            "relevant_articles": 5,
            "executive_summary": ["Key theme 1.", "Key theme 2."],
            "sections": [
                {
                    "impact_type": "Pricing",
                    "articles": [
                        {
                            "title": "Tariff Article",
                            "source": "Reuters",
                            "url": "https://example.com",
                            "published_date": "2026-05-15",
                            "relevance_score": 80,
                            "impact_level": "High",
                            "impact_type": "Pricing",
                            "summary": "A summary.",
                        }
                    ],
                }
            ],
            "recommended_actions": ["Action 1.", "Action 2."],
            "generated_at": "2026-05-19T10:00:00+00:00",
            "sources_used": ["Google News RSS"],
            "disclaimer": "Test disclaimer.",
        }

    def test_contains_category_name(self):
        md = _render_markdown(self._data())
        self.assertIn("Tires", md)

    def test_contains_article_title(self):
        md = _render_markdown(self._data())
        self.assertIn("Tariff Article", md)

    def test_contains_impact_section_header(self):
        md = _render_markdown(self._data())
        self.assertIn("Pricing", md)

    def test_contains_recommended_actions(self):
        md = _render_markdown(self._data())
        self.assertIn("Action 1.", md)

    def test_starts_with_h1(self):
        md = _render_markdown(self._data())
        self.assertTrue(md.startswith("# Category News Report"))


class TestRenderHTML(unittest.TestCase):

    def _data(self) -> dict:
        return {
            "category": "Tires",
            "date_start": "May 12, 2026",
            "date_end": "May 19, 2026",
            "articles_analyzed": 30,
            "relevant_articles": 5,
            "executive_summary": ["Theme 1."],
            "sections": [
                {
                    "impact_type": "Supply Chain",
                    "articles": [
                        {
                            "title": "Rubber Shortage",
                            "source": "Bloomberg",
                            "url": "https://example.com/rubber",
                            "published_date": "2026-05-15",
                            "relevance_score": 70,
                            "impact_level": "High",
                            "impact_type": "Supply Chain",
                            "summary": "Shortage summary.",
                        }
                    ],
                }
            ],
            "recommended_actions": ["Watch inventory."],
            "generated_at": "2026-05-19T10:00:00+00:00",
            "sources_used": ["Google News RSS"],
            "disclaimer": "Disclaimer.",
        }

    def test_is_valid_html(self):
        html = _render_html_inline(self._data())
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("</html>", html)

    def test_contains_article_title(self):
        html = _render_html_inline(self._data())
        self.assertIn("Rubber Shortage", html)

    def test_self_contained_no_external_links(self):
        html = _render_html_inline(self._data())
        # No external CSS/JS references
        self.assertNotIn('<link rel="stylesheet"', html)
        self.assertNotIn('<script src=', html)


class TestRenderJSON(unittest.TestCase):

    def test_valid_json(self):
        data = {
            "category": "Tires",
            "date_start": "May 12, 2026",
            "date_end": "May 19, 2026",
            "articles_analyzed": 10,
            "relevant_articles": 2,
            "executive_summary": [],
            "sections": [],
            "recommended_actions": [],
            "generated_at": "2026-05-19T10:00:00+00:00",
            "sources_used": [],
            "disclaimer": "",
        }
        raw = _render_json(data)
        parsed = json.loads(raw)
        self.assertEqual(parsed["category"], "Tires")

    def test_pretty_printed(self):
        raw = _render_json({"a": 1})
        self.assertIn("\n", raw)


class TestSaveReport(unittest.TestCase):

    def _data(self) -> dict:
        return {
            "category": "Tires",
            "date_start": "May 12, 2026",
            "date_end": "May 19, 2026",
            "articles_analyzed": 10,
            "relevant_articles": 2,
            "executive_summary": ["Theme."],
            "sections": [],
            "recommended_actions": ["Action."],
            "generated_at": "2026-05-19T10:00:00+00:00",
            "sources_used": ["RSS"],
            "disclaimer": "Disclaimer.",
        }

    def test_saves_markdown_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_report(self._data(), fmt="markdown", output_dir=tmpdir)
            self.assertTrue(path.exists())
            self.assertEqual(path.suffix, ".md")

    def test_saves_html_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_report(self._data(), fmt="html", output_dir=tmpdir)
            self.assertTrue(path.exists())
            self.assertEqual(path.suffix, ".html")

    def test_saves_json_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_report(self._data(), fmt="json", output_dir=tmpdir)
            self.assertTrue(path.exists())
            parsed = json.loads(path.read_text())
            self.assertEqual(parsed["category"], "Tires")

    def test_directory_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_report(self._data(), fmt="markdown", output_dir=tmpdir)
            # Should be output_dir/tires/YYYY-MM-DD/report.md
            parts = path.parts
            self.assertIn("tires", parts)
            self.assertEqual(path.name, "report.md")

    def test_invalid_format_raises(self):
        with self.assertRaises(ValueError):
            save_report(self._data(), fmt="pdf")


if __name__ == "__main__":
    unittest.main()
