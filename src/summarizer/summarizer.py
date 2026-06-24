"""
Article summarizer — generates 2-3 sentence sales-impact summaries via Claude.

Design decisions:
- The system prompt is stable across all batches → cached with cache_control.
- Up to 5 articles are batched per API call to minimise round trips.
- Token-aware batching keeps each request well under Haiku's context limit.
- A local file cache (keyed by URL hash) avoids re-summarizing the same article.
- Exponential-backoff retry handles rate limits and transient errors.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

import anthropic

from src.models.article import ImpactType, ScoredArticle, SummarizedArticle

logger = logging.getLogger(__name__)

# Approximate characters-per-token for rough budget estimation.
_CHARS_PER_TOKEN = 4
# Leave headroom below the model's context window.
_MAX_INPUT_TOKENS_PER_BATCH = 6_000
# Maximum articles per batch regardless of token estimate.
_MAX_ARTICLES_PER_BATCH = 5

_SYSTEM_PROMPT = (
    "You are a senior Amazon retail strategist with deep expertise in marketplace dynamics. "
    "For each article provided, write a 2-3 sentence analysis that:\n"
    "1. States the specific event and names the companies, products, or regulations involved.\n"
    "2. Provides a STRATEGIC insight — quantify the impact where possible (e.g. '15% tariff on imports from X', "
    "'affects ~$2B in annual category revenue', 'top 3 brands control 60% share'). "
    "Connect dots between the news and second-order effects on the Amazon marketplace: "
    "ASP shifts, buy box dynamics, seller behavior changes, search demand signals, or catalog gaps.\n\n"
    "AVOID generic statements like 'may affect sales', 'consider restocking', or 'monitor the situation'. "
    "Instead, be precise: WHO is affected, by HOW MUCH, over WHAT timeframe, and what the competitive implication is. "
    "Write like a Goldman Sachs analyst briefing a portfolio manager, not a news aggregator.\n\n"
    "Respond with a JSON object whose keys are the article indices (0-based integers as strings) "
    "and values are the summary strings. Example for two articles:\n"
    '{"0": "Summary of article 0.", "1": "Summary of article 1."}\n\n'
    "Respond with valid JSON only — no markdown fences, no explanation outside the JSON."
)

# Approximate token cost of the system prompt (used for budget calculation).
_SYSTEM_PROMPT_TOKENS = len(_SYSTEM_PROMPT) // _CHARS_PER_TOKEN


def _cache_key(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()


def _load_summary_cache(url: str, cache_dir: str) -> Optional[str]:
    path = Path(cache_dir) / "summaries" / f"{_cache_key(url)}.txt"
    if path.exists():
        logger.debug("Summary cache hit for %s", url[:60])
        return path.read_text(encoding="utf-8")
    return None


def _save_summary_cache(url: str, summary: str, cache_dir: str) -> None:
    dir_ = Path(cache_dir) / "summaries"
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / f"{_cache_key(url)}.txt").write_text(summary, encoding="utf-8")


def _article_block(index: int, scored: ScoredArticle) -> str:
    """Format a single article for inclusion in a batch prompt."""
    pub = scored.published_date
    pub_str = pub.strftime("%Y-%m-%d") if pub else "unknown"
    return (
        f"Article {index}:\n"
        f"Title: {scored.title}\n"
        f"Source: {scored.source} | Date: {pub_str}\n"
        f"Snippet: {scored.article.snippet or '(none)'}\n"
        f"Impact type: {scored.impact_type}\n"
    )


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _call_with_retry(
    client: anthropic.Anthropic,
    model: str,
    max_tokens: int,
    system: list[dict],
    messages: list[dict],
    max_retries: int,
    base_delay: float,
) -> anthropic.types.Message:
    for attempt in range(1, max_retries + 1):
        try:
            return client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
            )
        except anthropic.RateLimitError as exc:
            wait = base_delay * (2 ** (attempt - 1))
            if attempt < max_retries:
                logger.warning("Rate limit hit (attempt %d/%d) — retrying in %.1fs", attempt, max_retries, wait)
                time.sleep(wait)
            else:
                logger.error("Rate limit persisted after %d attempts", max_retries)
                raise
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                wait = base_delay * (2 ** (attempt - 1))
                if attempt < max_retries:
                    logger.warning("Server error %d (attempt %d/%d) — retrying in %.1fs", exc.status_code, attempt, max_retries, wait)
                    time.sleep(wait)
                    continue
            logger.error("Non-retryable API error: %s", exc)
            raise
    # Unreachable, but satisfies type checker
    raise RuntimeError("Retry loop exhausted without result or exception")


def _summarize_batch(
    batch: list[ScoredArticle],
    batch_start_index: int,
    client: anthropic.Anthropic,
    model: str,
    max_output_tokens: int,
    max_retries: int,
    base_delay: float,
) -> dict[int, str]:
    """
    Summarize a batch of articles in a single API call.

    Returns a dict mapping original list index → summary string.
    """
    user_content = "\n---\n".join(
        _article_block(batch_start_index + i, scored)
        for i, scored in enumerate(batch)
    )
    user_content += (
        f"\n\nPlease summarize articles {batch_start_index} through "
        f"{batch_start_index + len(batch) - 1} with their Amazon sales impact."
    )

    response = _call_with_retry(
        client=client,
        model=model,
        max_tokens=max_output_tokens,
        # Cache the stable system prompt across all batches.
        system=[{"type": "text", "text": _SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_content}],
        max_retries=max_retries,
        base_delay=base_delay,
    )

    usage = response.usage
    logger.debug(
        "Batch summarize usage — input: %d, cache_create: %d, cache_read: %d, output: %d",
        usage.input_tokens,
        getattr(usage, "cache_creation_input_tokens", 0),
        getattr(usage, "cache_read_input_tokens", 0),
        usage.output_tokens,
    )

    text = next(b.text for b in response.content if b.type == "text")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        logger.error("Summarizer returned non-JSON: %s", text[:300])
        # Fall back: assign the raw text as the summary for every article in the batch.
        return {batch_start_index + i: text.strip() for i in range(len(batch))}

    # Keys from the LLM are strings; convert to int and re-map to original indices.
    result: dict[int, str] = {}
    for i, scored in enumerate(batch):
        key = str(batch_start_index + i)
        summary = raw.get(key, "Summary unavailable.")
        result[batch_start_index + i] = summary.strip()
    return result


def _build_batches(
    to_summarize: list[tuple[int, ScoredArticle]],
) -> list[list[tuple[int, ScoredArticle]]]:
    """
    Group (index, article) pairs into token-aware batches.

    Each batch stays within _MAX_INPUT_TOKENS_PER_BATCH and _MAX_ARTICLES_PER_BATCH.
    """
    batches: list[list[tuple[int, ScoredArticle]]] = []
    current_batch: list[tuple[int, ScoredArticle]] = []
    current_tokens = _SYSTEM_PROMPT_TOKENS

    for idx, scored in to_summarize:
        block = _article_block(idx, scored)
        tokens = _estimate_tokens(block)
        if current_batch and (
            len(current_batch) >= _MAX_ARTICLES_PER_BATCH
            or current_tokens + tokens > _MAX_INPUT_TOKENS_PER_BATCH
        ):
            batches.append(current_batch)
            current_batch = []
            current_tokens = _SYSTEM_PROMPT_TOKENS
        current_batch.append((idx, scored))
        current_tokens += tokens

    if current_batch:
        batches.append(current_batch)
    return batches


_PERCENT_RE = re.compile(r'\d+\.?\d*\s*%')
_DOLLAR_RE = re.compile(r'\$[\d,]+\.?\d*\s*(?:billion|million|B|M)?', re.IGNORECASE)
_NUMBER_RE = re.compile(r'\b\d[\d,]*\.?\d*\s*(?:billion|million|thousand|units|stores|locations)?\b', re.IGNORECASE)


def _extract_numbers(text: str) -> list[str]:
    """Pull out percentages, dollar amounts, and significant numbers."""
    found = []
    found.extend(_PERCENT_RE.findall(text))
    found.extend(_DOLLAR_RE.findall(text))
    if not found:
        nums = _NUMBER_RE.findall(text)
        found.extend(n for n in nums if any(w in n.lower() for w in ("billion", "million", "thousand", "units", "stores")))
    return found[:3]


def _extract_entities(text: str, brands: list[str]) -> list[str]:
    """Find brand/company names that appear in the text."""
    found = []
    text_lower = text.lower()
    for brand in brands:
        if brand.lower() in text_lower:
            found.append(brand)
    return found[:4]


def _template_summary(scored: ScoredArticle) -> str:
    """
    Generate a non-LLM summary by extracting signal from the article text.

    Pulls out named entities, numbers, and builds a specific impact statement
    based on what's actually in the article rather than generic framing.
    """
    snippet = (scored.article.snippet or "").strip()
    title = scored.title
    full_text = f"{title} {snippet}"

    # Extract the core event (first 2 sentences of snippet, or title)
    if snippet:
        sentences = re.split(r'(?<=[.!?])\s+', snippet)
        what_happened = " ".join(sentences[:2]).strip()
        if not what_happened.endswith((".", "!", "?")):
            what_happened += "."
    else:
        what_happened = title.rstrip(".!?") + "."

    # Extract quantitative data
    numbers = _extract_numbers(full_text)
    number_context = ""
    if numbers:
        number_context = f" Key figures: {', '.join(numbers[:2])}."

    # Build impact statement based on what's actually in the article
    impact_signals = []

    if scored.impact_type == "Supply Chain":
        if any(w in full_text.lower() for w in ("shortage", "delay", "halt", "closure", "shut")):
            impact_signals.append("Expect constrained supply and potential stockouts on affected SKUs")
        elif any(w in full_text.lower() for w in ("recall", "safety")):
            impact_signals.append("Listings may require removal or compliance updates")
        else:
            impact_signals.append("Supply-side disruption that could tighten inventory availability")

    elif scored.impact_type == "Pricing":
        if any(w in full_text.lower() for w in ("tariff", "duty", "import tax")):
            impact_signals.append("Import cost increase will pressure ASPs upward across affected brands")
        elif any(w in full_text.lower() for w in ("price war", "price cut", "discount")):
            impact_signals.append("Downward price pressure — expect margin compression and buy box volatility")
        else:
            impact_signals.append("Pricing dynamics shifting — review ASP trends on affected ASINs")

    elif scored.impact_type == "Demand":
        if any(w in full_text.lower() for w in ("viral", "trending", "surge", "record")):
            impact_signals.append("Demand spike likely — sellers without stock positioned will lose share")
        elif any(w in full_text.lower() for w in ("decline", "slowdown", "drop")):
            impact_signals.append("Demand softening — overexposed sellers risk excess inventory and markdowns")
        else:
            impact_signals.append("Consumer demand signal detected — watch search volume trends this week")

    elif scored.impact_type == "Regulatory":
        if any(w in full_text.lower() for w in ("ban", "banned", "prohibit")):
            impact_signals.append("Product ban could force delisting — check affected ASINs immediately")
        elif any(w in full_text.lower() for w in ("recall")):
            impact_signals.append("Recall action may require immediate listing suppression and seller notification")
        else:
            impact_signals.append("Regulatory change — audit compliance of affected listings before enforcement")

    elif scored.impact_type == "Competitive":
        if any(w in full_text.lower() for w in ("bankrupt", "closing", "layoff", "restructur")):
            impact_signals.append("Competitor weakening — opportunity to capture displaced demand and market share")
        elif any(w in full_text.lower() for w in ("launch", "new product", "partnership")):
            impact_signals.append("New competitive threat entering — assess overlap with top-selling ASINs")
        else:
            impact_signals.append("Competitive landscape shifting — review relative positioning of key brands")

    elif scored.impact_type == "Seasonal":
        if any(w in full_text.lower() for w in ("prime day", "prime week", "black friday", "holiday")):
            impact_signals.append("Event-driven demand window — ensure inventory depth and promotional alignment")
        else:
            impact_signals.append("Seasonal trigger detected — validate forecast and inventory positioning")

    else:
        impact_signals.append("Review for direct category impact")

    impact = impact_signals[0] if impact_signals else ""
    return f"{what_happened}{number_context} {impact}."


def summarize_articles(
    scored_articles: list[ScoredArticle],
    category: str,
    client: Optional[anthropic.Anthropic] = None,
    model: str = "claude-haiku-4-5",
    max_output_tokens: int = 1024,
    max_retries: int = 3,
    base_delay: float = 1.0,
    cache_dir: str = ".cache",
) -> list[SummarizedArticle]:
    """
    Generate sales-impact summaries for a list of scored articles.

    - Articles already in cache are not re-summarized.
    - If *client* is None, uses template-based fallback summaries (no API call).
    - Otherwise, articles are batched (≤5 per call) and summarized via Claude.
    - Returns SummarizedArticle objects in the same order as input.
    """
    summaries: dict[int, str] = {}
    to_summarize: list[tuple[int, ScoredArticle]] = []

    # Check cache first (cache works in both LLM and template modes)
    for i, scored in enumerate(scored_articles):
        cached = _load_summary_cache(scored.url, cache_dir)
        if cached is not None:
            summaries[i] = cached
        else:
            to_summarize.append((i, scored))

    if client is None:
        # ── Template fallback ─────────────────────────────────────────────────
        logger.info(
            "No Anthropic client — using template summaries for %d new articles "
            "(%d from cache) for category %r",
            len(to_summarize), len(summaries), category,
        )
        for i, scored in to_summarize:
            summaries[i] = _template_summary(scored)
            # Don't cache template summaries — we want them replaced by LLM ones
            # the moment a key becomes available.
    else:
        # ── LLM path ──────────────────────────────────────────────────────────
        logger.info(
            "Summarizing %d new articles (%d from cache) for category %r via Claude",
            len(to_summarize), len(summaries), category,
        )
        if to_summarize:
            batches = _build_batches(to_summarize)
            logger.info("Sending %d batch(es) to Claude", len(batches))

            for batch in batches:
                batch_indices = [idx for idx, _ in batch]
                batch_articles = [a for _, a in batch]
                batch_start = batch_indices[0]

                try:
                    batch_summaries = _summarize_batch(
                        batch=batch_articles,
                        batch_start_index=batch_start,
                        client=client,
                        model=model,
                        max_output_tokens=max_output_tokens,
                        max_retries=max_retries,
                        base_delay=base_delay,
                    )
                    for orig_idx, summary in batch_summaries.items():
                        summaries[orig_idx] = summary
                        _save_summary_cache(scored_articles[orig_idx].url, summary, cache_dir)
                except Exception as exc:
                    logger.error("Batch summarization failed: %s — falling back to templates", exc)
                    for idx, scored in batch:
                        summaries[idx] = _template_summary(scored)

    return [
        SummarizedArticle(scored=scored_articles[i], summary=summaries.get(i, "Summary unavailable."))
        for i in range(len(scored_articles))
    ]


def group_summarized_by_impact(
    summarized: list[SummarizedArticle],
) -> dict[ImpactType, list[SummarizedArticle]]:
    """Group summarized articles by impact type, each group sorted by score descending."""
    groups: dict[ImpactType, list[SummarizedArticle]] = {}
    for article in summarized:
        groups.setdefault(article.impact_type, []).append(article)
    for group in groups.values():
        group.sort(key=lambda a: a.relevance_score, reverse=True)
    return groups
