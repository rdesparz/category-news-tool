"""Category keyword mapping with LLM fallback for unknown categories."""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import anthropic
import yaml

logger = logging.getLogger(__name__)

# System prompt is stable across all calls — cache it.
_SYSTEM_PROMPT = """You are a retail category intelligence expert specializing in Amazon GL (General Ledger) product categories.
Your task is to generate comprehensive keyword sets for news monitoring of product categories.

For each category, produce keywords in five groups:
- primary: core product terms (3-6 terms)
- brands: major brand names in the space (5-10 brands)
- supply_chain: logistics, pricing, sourcing, recall, shortage terms (4-8 terms)
- seasonal: trend, usage-context, and seasonal terms (4-8 terms)
- regulatory: standards bodies, compliance, labeling, safety regulations (3-6 terms)

Respond with valid JSON only — no markdown, no explanation."""


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_category_config(category: str, config_dir: str = "data/category_configs") -> Optional[dict]:
    """Return keyword dict from YAML if it exists, else None."""
    slug = category.lower().replace(" ", "_")
    path = Path(config_dir) / f"{slug}.yaml"
    if not path.exists():
        logger.debug("No YAML config found for %r at %s", category, path)
        return None
    with open(path) as f:
        data = yaml.safe_load(f)
    logger.info("Loaded keyword config for %r from %s", category, path)
    return data.get("keywords", {})


def generate_keywords_via_llm(
    category: str,
    client: anthropic.Anthropic,
    model: str = "claude-haiku-4-5",
    max_tokens: int = 1024,
) -> dict:
    """
    Call Claude Haiku to generate keywords for an unmapped category.

    Uses prompt caching on the stable system prompt so repeated calls
    for different categories pay the cache-read rate instead of full price.
    """
    logger.info("Generating keywords for %r via LLM (%s)", category, model)

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        # Cache the system prompt — it is identical across every category call.
        system=[
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": (
                    f'Generate news-monitoring keywords for the Amazon product category: "{category}"\n\n'
                    "Return JSON with keys: primary, brands, supply_chain, seasonal, regulatory. "
                    "Each value is a list of strings."
                ),
            }
        ],
    )

    # Log cache usage so callers can verify hits on repeated runs.
    usage = response.usage
    logger.debug(
        "LLM usage — input: %d, cache_create: %d, cache_read: %d, output: %d",
        usage.input_tokens,
        getattr(usage, "cache_creation_input_tokens", 0),
        getattr(usage, "cache_read_input_tokens", 0),
        usage.output_tokens,
    )

    text = next(b.text for b in response.content if b.type == "text")
    try:
        keywords = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error("LLM returned non-JSON for %r: %s", category, text[:200])
        raise ValueError(f"LLM did not return valid JSON for category {category!r}") from exc

    _validate_keyword_structure(keywords, category)
    return keywords


def _validate_keyword_structure(keywords: dict, category: str) -> None:
    expected = {"primary", "brands", "supply_chain", "seasonal", "regulatory"}
    missing = expected - keywords.keys()
    if missing:
        logger.warning("LLM keywords for %r missing groups: %s", category, missing)
    for key, value in keywords.items():
        if not isinstance(value, list):
            raise ValueError(f"Keyword group {key!r} for {category!r} must be a list, got {type(value)}")


def get_keywords(
    category: str,
    config: Optional[dict] = None,
    config_dir: str = "data/category_configs",
    anthropic_client: Optional[anthropic.Anthropic] = None,
) -> dict:
    """
    Return keyword dict for a category.

    1. Try loading from a YAML config file.
    2. Fall back to LLM generation if not found.
    """
    keywords = load_category_config(category, config_dir)
    if keywords:
        return keywords

    logger.info("No pre-mapped keywords for %r — falling back to LLM", category)
    if anthropic_client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        anthropic_client = anthropic.Anthropic(api_key=api_key)

    model = "claude-haiku-4-5"
    max_tokens = 1024
    if config:
        model = config.get("anthropic", {}).get("model", model)
        max_tokens = config.get("anthropic", {}).get("max_tokens", max_tokens)

    return generate_keywords_via_llm(category, anthropic_client, model=model, max_tokens=max_tokens)


def flatten_keywords(keywords: dict) -> list[str]:
    """Return a deduplicated flat list of all keyword strings."""
    seen = set()
    flat = []
    for terms in keywords.values():
        for term in terms:
            if term not in seen:
                seen.add(term)
                flat.append(term)
    return flat
