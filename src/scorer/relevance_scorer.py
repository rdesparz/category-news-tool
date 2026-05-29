"""
Relevance scorer for category news articles.

Scores each article 0-100 based on keyword signals across seven dimensions.
Articles below the configured threshold are filtered out.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from src.models.article import (
    Article,
    ImpactType,
    ScoreBreakdown,
    ScoredArticle,
)

logger = logging.getLogger(__name__)

# ── Keyword banks for signal dimensions ────────────────────────────────────
# These supplement the per-category keyword config with universal signals.

_SUPPLY_CHAIN_SIGNALS = {
    "shortage", "shortages", "supply chain", "supply-chain", "disruption",
    "disruptions", "delay", "delays", "factory closure", "plant shutdown",
    "logistics", "backlog", "inventory", "out of stock", "stockout",
    "port congestion", "freight", "raw material", "raw materials",
    "production halt", "recall", "supplier",
    # "import"/"export" omitted — too generic; appears in tariff articles
}

_PRICING_SIGNALS = {
    "tariff", "tariffs", "price increase", "price hike", "price war",
    "inflation", "cost increase", "surcharge", "levy", "duty", "duties",
    "sanctions", "trade war", "import tax", "anti-dumping", "cost pressure",
    "price cut", "discounting", "margin pressure",
}

_DEMAND_SIGNALS = {
    "demand surge", "demand spike", "trending", "viral", "consumer demand",
    "sales growth", "adoption", "growing popularity", "record sales",
    "boom", "surge in demand", "new technology", "ev", "electric vehicle",
    "sustainability", "green", "consumer preference", "shift",
}

_REGULATORY_SIGNALS = {
    "recall", "regulation", "regulations", "regulatory", "mandate", "mandates",
    "compliance", "standard", "standards", "ban", "banned", "law", "legislation",
    "nhtsa", "fda", "epa", "dot", "cpsc", "safety requirement", "labeling",
    "certification", "penalty", "fine", "enforcement",
}

_SEASONAL_SIGNALS = {
    "winter", "summer", "spring", "fall", "autumn", "holiday", "christmas",
    "black friday", "prime day", "seasonal", "weather", "storm", "hurricane",
    "flood", "snow", "ice", "back to school", "new year",
}

_COMPETITOR_SIGNALS = {
    "bankruptcy", "bankrupt", "acquisition", "merger", "acquired", "ipo",
    "earnings", "revenue", "market share", "launches", "launch", "new product",
    "partnership", "joint venture", "layoff", "restructuring", "plant expansion",
}


def _keyword_hit(text: str, terms: set[str]) -> bool:
    """True if any term in *terms* appears in *text* (leading word boundary, plurals included)."""
    for term in terms:
        # Leading word boundary only — 'import' matches 'imports', 'shortage' matches 'shortages'.
        if re.search(r'\b' + re.escape(term), text):
            return True
    return False


def _primary_hit(text: str, primary_keywords: list[str]) -> bool:
    """True if any of the category's primary product keywords are in the text."""
    for kw in primary_keywords:
        if re.search(r'\b' + re.escape(kw.lower()) + r'\b', text):
            return True
    return False


def _brand_hit(text: str, brand_keywords: list[str]) -> bool:
    """True if any brand name appears in the text (case-insensitive)."""
    for brand in brand_keywords:
        if brand.lower() in text:
            return True
    return False


def _category_supply_chain_hit(text: str, supply_chain_keywords: list[str]) -> bool:
    for kw in supply_chain_keywords:
        if kw.lower() in text:
            return True
    return False


def _category_seasonal_hit(text: str, seasonal_keywords: list[str]) -> bool:
    for kw in seasonal_keywords:
        if kw.lower() in text:
            return True
    return False


def _category_regulatory_hit(text: str, regulatory_keywords: list[str]) -> bool:
    for kw in regulatory_keywords:
        if kw.lower() in text:
            return True
    return False


# ── Impact-type detection ───────────────────────────────────────────────────

def _dominant_impact_type(breakdown: ScoreBreakdown) -> ImpactType:
    """
    Pick the single highest-scoring *specific* dimension as the impact type.

    direct_mention is excluded from the competition — it represents "the article
    is about this product" rather than a distinctive impact vector, so it acts only
    as a score booster.  If no specific dimension fired, return "General".
    """
    scores: list[tuple[int, ImpactType]] = [
        (breakdown.supply_chain, "Supply Chain"),
        (breakdown.pricing_tariff, "Pricing"),
        (breakdown.consumer_demand, "Demand"),
        (breakdown.regulatory, "Regulatory"),
        (breakdown.competitor_brand, "Competitive"),
        (breakdown.seasonal, "Seasonal"),
    ]
    scores.sort(key=lambda x: x[0], reverse=True)
    if scores[0][0] == 0:
        return "General"
    return scores[0][1]


# ── Core scoring ─────────────────────────────────────────────────────────────

def score_article(
    article: Article,
    keywords: dict,
    threshold: int = 30,
) -> Optional[ScoredArticle]:
    """
    Score a single article against the category keyword set.

    Returns None if the score is below *threshold*.

    Scoring rubric (additive, capped per dimension):
      +30  direct product mention
      +25  supply chain impact
      +25  pricing / tariff impact
      +20  consumer demand signal
      +15  competitor / brand news
      +20  regulatory change
      +10  seasonal relevance
    """
    text = article.text_for_scoring()

    primary_kws: list[str] = keywords.get("primary", [])
    brand_kws: list[str] = keywords.get("brands", [])
    supply_kws: list[str] = keywords.get("supply_chain", [])
    seasonal_kws: list[str] = keywords.get("seasonal", [])
    regulatory_kws: list[str] = keywords.get("regulatory", [])

    bd = ScoreBreakdown()

    # +30 — direct product mention
    if _primary_hit(text, primary_kws):
        bd.direct_mention = 30

    # +25 — supply chain (category-specific terms take priority, then universal)
    if _category_supply_chain_hit(text, supply_kws) or _keyword_hit(text, _SUPPLY_CHAIN_SIGNALS):
        bd.supply_chain = 25

    # +25 — pricing / tariff
    if _keyword_hit(text, _PRICING_SIGNALS):
        bd.pricing_tariff = 25

    # +20 — consumer demand
    if _keyword_hit(text, _DEMAND_SIGNALS):
        bd.consumer_demand = 20

    # +15 — competitor / brand news
    if _brand_hit(text, brand_kws) or _keyword_hit(text, _COMPETITOR_SIGNALS):
        bd.competitor_brand = 15

    # +20 — regulatory
    if _category_regulatory_hit(text, regulatory_kws) or _keyword_hit(text, _REGULATORY_SIGNALS):
        bd.regulatory = 20

    # +10 — seasonal
    if _category_seasonal_hit(text, seasonal_kws) or _keyword_hit(text, _SEASONAL_SIGNALS):
        bd.seasonal = 10

    total = min(bd.total, 100)

    if total < threshold:
        logger.debug("Filtered out %r (score=%d)", article.title[:60], total)
        return None

    if total >= 70:
        level = "High"
    elif total >= 40:
        level = "Medium"
    else:
        level = "Low"

    impact_type = _dominant_impact_type(bd)

    return ScoredArticle(
        article=article,
        relevance_score=total,
        impact_level=level,
        impact_type=impact_type,
        score_breakdown=bd,
    )


def score_articles(
    articles: list[Article],
    keywords: dict,
    threshold: int = 30,
) -> list[ScoredArticle]:
    """Score and filter a list of articles, returning only those above threshold."""
    scored = []
    for article in articles:
        result = score_article(article, keywords, threshold)
        if result is not None:
            scored.append(result)

    scored.sort(key=lambda a: a.relevance_score, reverse=True)
    logger.info(
        "Scored %d articles: %d passed threshold=%d",
        len(articles), len(scored), threshold,
    )
    return scored


def group_by_impact_type(scored: list[ScoredArticle]) -> dict[ImpactType, list[ScoredArticle]]:
    """Group scored articles by their dominant impact type."""
    groups: dict[ImpactType, list[ScoredArticle]] = {}
    for article in scored:
        groups.setdefault(article.impact_type, []).append(article)
    return groups
