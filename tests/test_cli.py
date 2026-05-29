"""Tests for the CLI interface."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

sys.path.insert(0, str(Path(__file__).parent.parent))

from main import _add_category, cli


class TestListCategories(unittest.TestCase):

    def test_lists_configured_categories(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("data/category_configs").mkdir(parents=True)
            Path("data/category_configs/tires.yaml").write_text(
                "category: Tires\nkeywords:\n  primary: [tire]\n  brands: []\n  supply_chain: []\n  seasonal: []\n  regulatory: []\n"
            )
            Path("config.yaml").write_text(
                "anthropic:\n  model: claude-haiku-4-5\n  max_tokens: 512\n"
                "fetcher:\n  days_lookback: 7\n  dedup_threshold: 85\n  cache_ttl_hours: 6\n"
                "  cache_dir: .cache\n  request_timeout: 5\n  max_retries: 1\n  retry_base_delay: 0\n"
                "scorer:\n  threshold: 30\nsummarizer:\n  max_articles_per_batch: 5\n"
                "  max_output_tokens: 512\n  max_retries: 1\n  retry_base_delay: 0\n"
                "report:\n  output_dir: reports\n  default_format: markdown\n"
                "logging:\n  level: WARNING\n  format: '%(message)s'\n"
            )
            result = runner.invoke(cli, ["--list-categories"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Tires", result.output)


class TestAddCategory(unittest.TestCase):

    def test_add_category_creates_yaml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cat_dir = Path(tmpdir) / "data" / "category_configs"
            cat_dir.mkdir(parents=True)
            _original_dir = "data/category_configs"

            import main as _main
            orig = _main._CATEGORY_CONFIG_DIR
            _main._CATEGORY_CONFIG_DIR = str(cat_dir)
            try:
                _add_category("Pet Food", "dog food,cat food,Purina,Blue Buffalo,pet food recall")
            finally:
                _main._CATEGORY_CONFIG_DIR = orig

            yaml_path = cat_dir / "pet_food.yaml"
            self.assertTrue(yaml_path.exists())

    def test_add_category_yaml_content(self):
        import yaml as _yaml

        with tempfile.TemporaryDirectory() as tmpdir:
            cat_dir = Path(tmpdir) / "configs"
            cat_dir.mkdir()
            import main as _main
            orig = _main._CATEGORY_CONFIG_DIR
            _main._CATEGORY_CONFIG_DIR = str(cat_dir)
            try:
                _add_category("Pet Food", "dog food,cat food,Purina,pet food recall")
            finally:
                _main._CATEGORY_CONFIG_DIR = orig

            data = _yaml.safe_load((cat_dir / "pet_food.yaml").read_text())
            self.assertEqual(data["category"], "Pet Food")
            self.assertIn("keywords", data)

    def test_add_category_cli_flag(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("data/category_configs").mkdir(parents=True)
            Path("config.yaml").write_text(
                "anthropic:\n  model: claude-haiku-4-5\n  max_tokens: 512\n"
                "fetcher:\n  days_lookback: 7\n  dedup_threshold: 85\n  cache_ttl_hours: 6\n"
                "  cache_dir: .cache\n  request_timeout: 5\n  max_retries: 1\n  retry_base_delay: 0\n"
                "scorer:\n  threshold: 30\nsummarizer:\n  max_articles_per_batch: 5\n"
                "  max_output_tokens: 512\n  max_retries: 1\n  retry_base_delay: 0\n"
                "report:\n  output_dir: reports\n  default_format: markdown\n"
                "logging:\n  level: WARNING\n  format: '%(message)s'\n"
            )
            result = runner.invoke(
                cli, ["--add-category", "Coffee Makers", "--keywords", "coffee,espresso,Nespresso,Keurig"]
            )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Coffee Makers", result.output)

    def test_add_category_without_keywords_errors(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("config.yaml").write_text(
                "anthropic:\n  model: claude-haiku-4-5\n  max_tokens: 512\n"
                "fetcher:\n  days_lookback: 7\n  dedup_threshold: 85\n  cache_ttl_hours: 6\n"
                "  cache_dir: .cache\n  request_timeout: 5\n  max_retries: 1\n  retry_base_delay: 0\n"
                "scorer:\n  threshold: 30\nsummarizer:\n  max_articles_per_batch: 5\n"
                "  max_output_tokens: 512\n  max_retries: 1\n  retry_base_delay: 0\n"
                "report:\n  output_dir: reports\n  default_format: markdown\n"
                "logging:\n  level: WARNING\n  format: '%(message)s'\n"
            )
            result = runner.invoke(cli, ["--add-category", "Coffee Makers"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--keywords", result.output)


class TestFullPipeline(unittest.TestCase):
    """Integration-style tests that mock all external I/O."""

    def _config_text(self) -> str:
        return (
            "anthropic:\n  model: claude-haiku-4-5\n  max_tokens: 512\n"
            "newsapi:\n  api_key: ''\n  base_url: https://newsapi.org/v2\n"
            "gnews:\n  api_key: ''\n  base_url: https://gnews.io/api/v4\n"
            "fetcher:\n  days_lookback: 7\n  dedup_threshold: 85\n  cache_ttl_hours: 6\n"
            "  cache_dir: .cache\n  request_timeout: 5\n  max_retries: 1\n  retry_base_delay: 0\n"
            "scorer:\n  threshold: 30\nsummarizer:\n  max_articles_per_batch: 5\n"
            "  max_output_tokens: 512\n  max_retries: 1\n  retry_base_delay: 0\n"
            "report:\n  output_dir: reports\n  default_format: markdown\n"
            "logging:\n  level: WARNING\n  format: '%(message)s'\n"
        )

    def _mock_anthropic_client(self) -> MagicMock:
        """Client that returns valid text for any summarize/exec-summary call."""
        block = MagicMock()
        block.type = "text"
        block.text = '{"0": "Tariff will increase tire prices on Amazon."}'
        usage = MagicMock()
        usage.input_tokens = 80
        usage.cache_creation_input_tokens = 20
        usage.cache_read_input_tokens = 0
        usage.output_tokens = 40
        msg = MagicMock()
        msg.content = [block]
        msg.usage = usage
        client = MagicMock(spec=["messages"])
        client.messages = MagicMock()
        client.messages.create.return_value = msg
        return client

    def _stub_exec_responses(self, client: MagicMock) -> None:
        """Make subsequent calls (exec summary, actions) return bullet text."""
        bullet_block = MagicMock()
        bullet_block.type = "text"
        bullet_block.text = "- Key insight this week."
        bullet_usage = MagicMock()
        bullet_usage.input_tokens = 50
        bullet_usage.cache_creation_input_tokens = 10
        bullet_usage.cache_read_input_tokens = 0
        bullet_usage.output_tokens = 20
        bullet_msg = MagicMock()
        bullet_msg.content = [bullet_block]
        bullet_msg.usage = bullet_usage

        summary_block = MagicMock()
        summary_block.type = "text"
        summary_block.text = '{"0": "Tariff impact summary."}'
        summary_usage = MagicMock()
        summary_usage.input_tokens = 80
        summary_usage.cache_creation_input_tokens = 20
        summary_usage.cache_read_input_tokens = 0
        summary_usage.output_tokens = 40
        summary_msg = MagicMock()
        summary_msg.content = [summary_block]
        summary_msg.usage = summary_usage

        # summarize call returns JSON, exec/action calls return bullets
        client.messages.create.side_effect = [
            summary_msg,   # article summarization
            bullet_msg,    # executive summary
            bullet_msg,    # recommended actions
        ]

    def test_single_category_markdown(self):

        raw_articles = [
            {
                "title": "US Announces 25% Tariff on Chinese Tire Imports",
                "source": "Reuters",
                "url": "https://example.com/tariff",
                "published_date": datetime(2026, 5, 15, tzinfo=timezone.utc),
                "snippet": "Goodyear warns tariff on tire imports will raise prices.",
            }
        ]

        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("data/category_configs").mkdir(parents=True)
            Path("data/category_configs/tires.yaml").write_text(
                "category: Tires\nkeywords:\n  primary: [tire, tires]\n"
                "  brands: [Goodyear, Michelin]\n  supply_chain: [tire shortage]\n"
                "  seasonal: [winter tires]\n  regulatory: [NHTSA]\n"
            )
            Path("config.yaml").write_text(self._config_text())
            Path("reports").mkdir()
            Path(".cache").mkdir()

            client = self._mock_anthropic_client()
            self._stub_exec_responses(client)

            with (
                patch("main._get_anthropic_client", return_value=client),
                patch("main.fetch_news_for_category", return_value=raw_articles),
                patch("main.get_keywords", return_value={
                    "primary": ["tire", "tires"],
                    "brands": ["Goodyear"],
                    "supply_chain": [],
                    "seasonal": [],
                    "regulatory": [],
                }),
            ):
                result = runner.invoke(cli, ["--category", "Tires", "--format", "markdown"])

            self.assertEqual(result.exit_code, 0, result.output)
            report_files = list(Path("reports").glob("**/*.md"))
            self.assertTrue(len(report_files) >= 1, "No markdown report file was created")

    def test_json_format_produces_valid_json(self):

        raw_articles = [
            {
                "title": "Tire tariff announced",
                "source": "Bloomberg",
                "url": "https://example.com/t",
                "published_date": datetime(2026, 5, 15, tzinfo=timezone.utc),
                "snippet": "Goodyear tariff on tire imports.",
            }
        ]

        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("data/category_configs").mkdir(parents=True)
            Path("config.yaml").write_text(self._config_text())
            Path("reports").mkdir()
            Path(".cache").mkdir()

            client = self._mock_anthropic_client()
            self._stub_exec_responses(client)

            with (
                patch("main._get_anthropic_client", return_value=client),
                patch("main.fetch_news_for_category", return_value=raw_articles),
                patch("main.get_keywords", return_value={
                    "primary": ["tire"],
                    "brands": ["Goodyear"],
                    "supply_chain": [],
                    "seasonal": [],
                    "regulatory": [],
                }),
            ):
                runner.invoke(cli, ["--category", "Tires", "--format", "json"])

            json_files = list(Path("reports").glob("**/*.json"))
            self.assertTrue(len(json_files) >= 1)
            parsed = json.loads(json_files[0].read_text())
            self.assertIn("category", parsed)

    def test_missing_api_key_exits_cleanly(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("config.yaml").write_text(self._config_text())
            env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
            result = runner.invoke(cli, ["--category", "Tires"], env=env, catch_exceptions=False)
        # Should exit with non-zero and print a helpful error
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("ANTHROPIC_API_KEY", result.output)


if __name__ == "__main__":
    unittest.main()
