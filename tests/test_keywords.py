"""Tests for keyword loading and LLM fallback."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.keywords.category_keywords import (
    flatten_keywords,
    generate_keywords_via_llm,
    get_keywords,
    load_category_config,
)


class TestLoadCategoryConfig(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_loads_existing_yaml(self):
        yaml_content = """
category: Gadgets
keywords:
  primary: ["gadget", "device"]
  brands: ["Apple", "Samsung"]
  supply_chain: ["chip shortage"]
  seasonal: ["holiday"]
  regulatory: ["FCC"]
"""
        (Path(self.tmpdir) / "gadgets.yaml").write_text(yaml_content)
        result = load_category_config("Gadgets", config_dir=self.tmpdir)
        self.assertEqual(result["primary"], ["gadget", "device"])
        self.assertEqual(result["brands"], ["Apple", "Samsung"])

    def test_returns_none_for_missing_category(self):
        result = load_category_config("NonExistent", config_dir=self.tmpdir)
        self.assertIsNone(result)

    def test_slug_normalisation(self):
        yaml_content = "category: Winter Boots\nkeywords:\n  primary: [boots]\n  brands: []\n  supply_chain: []\n  seasonal: []\n  regulatory: []\n"
        (Path(self.tmpdir) / "winter_boots.yaml").write_text(yaml_content)
        result = load_category_config("Winter Boots", config_dir=self.tmpdir)
        self.assertIsNotNone(result)


class TestGenerateKeywordsViaLLM(unittest.TestCase):
    def _make_mock_client(self, keywords: dict):
        response_text = json.dumps(keywords)
        block = MagicMock()
        block.type = "text"
        block.text = response_text

        usage = MagicMock()
        usage.input_tokens = 50
        usage.cache_creation_input_tokens = 10
        usage.cache_read_input_tokens = 0
        usage.output_tokens = 100

        message = MagicMock()
        message.content = [block]
        message.usage = usage

        client = MagicMock()
        client.messages.create.return_value = message
        return client

    def test_returns_keyword_dict(self):
        expected = {
            "primary": ["widget"],
            "brands": ["BrandA"],
            "supply_chain": ["supply"],
            "seasonal": ["summer"],
            "regulatory": ["ISO"],
        }
        client = self._make_mock_client(expected)
        result = generate_keywords_via_llm("Widgets", client)
        self.assertEqual(result["primary"], ["widget"])

    def test_calls_api_with_cache_control(self):
        expected = {
            "primary": ["x"],
            "brands": ["B"],
            "supply_chain": ["s"],
            "seasonal": ["q"],
            "regulatory": ["r"],
        }
        client = self._make_mock_client(expected)
        generate_keywords_via_llm("TestCat", client)

        call_kwargs = client.messages.create.call_args.kwargs
        system = call_kwargs["system"]
        self.assertIsInstance(system, list)
        self.assertEqual(system[0]["cache_control"]["type"], "ephemeral")

    def test_raises_on_invalid_json(self):
        block = MagicMock()
        block.type = "text"
        block.text = "not json at all"
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

        with self.assertRaises(ValueError):
            generate_keywords_via_llm("BadCat", client)


class TestGetKeywords(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_prefers_yaml_over_llm(self):
        yaml_content = "category: Tires\nkeywords:\n  primary: [tire]\n  brands: []\n  supply_chain: []\n  seasonal: []\n  regulatory: []\n"
        (Path(self.tmpdir) / "tires.yaml").write_text(yaml_content)

        with patch("src.keywords.category_keywords.generate_keywords_via_llm") as mock_llm:
            result = get_keywords("Tires", config_dir=self.tmpdir)
            mock_llm.assert_not_called()
        self.assertEqual(result["primary"], ["tire"])

    def test_falls_back_to_llm_when_no_yaml(self):
        expected = {
            "primary": ["drone"],
            "brands": ["DJI"],
            "supply_chain": ["battery"],
            "seasonal": ["holiday"],
            "regulatory": ["FAA"],
        }
        with patch("src.keywords.category_keywords.generate_keywords_via_llm", return_value=expected) as mock_llm:
            result = get_keywords("Drones", config_dir=self.tmpdir)
            mock_llm.assert_called_once()
        self.assertEqual(result["primary"], ["drone"])


class TestFlattenKeywords(unittest.TestCase):
    def test_flattens_and_deduplicates(self):
        keywords = {
            "primary": ["tire", "wheel"],
            "brands": ["Michelin", "tire"],  # 'tire' is a dup
            "supply_chain": ["shortage"],
            "seasonal": ["winter tires"],
            "regulatory": ["NHTSA"],
        }
        flat = flatten_keywords(keywords)
        self.assertEqual(flat.count("tire"), 1)
        self.assertIn("Michelin", flat)
        self.assertIn("NHTSA", flat)


if __name__ == "__main__":
    unittest.main()
