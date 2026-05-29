"""Shared data models for the Category News Intelligence Tool."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional

ImpactLevel = Literal["High", "Medium", "Low"]
ImpactType = Literal[
    "Supply Chain",
    "Pricing",
    "Demand",
    "Regulatory",
    "Competitive",
    "Seasonal",
    "General",
]


@dataclass
class Article:
    """Raw article as returned by the fetcher."""

    title: str
    source: str
    url: str
    published_date: Optional[datetime]
    snippet: str

    def text_for_scoring(self) -> str:
        """Return the concatenated text used for keyword matching."""
        return f"{self.title} {self.snippet}".lower()


@dataclass
class ScoreBreakdown:
    direct_mention: int = 0        # max 30
    supply_chain: int = 0          # max 25
    pricing_tariff: int = 0        # max 25
    consumer_demand: int = 0       # max 20
    competitor_brand: int = 0      # max 15
    regulatory: int = 0            # max 20
    seasonal: int = 0              # max 10

    @property
    def total(self) -> int:
        return (
            self.direct_mention
            + self.supply_chain
            + self.pricing_tariff
            + self.consumer_demand
            + self.competitor_brand
            + self.regulatory
            + self.seasonal
        )


@dataclass
class ScoredArticle:
    """Article enriched with relevance score and impact metadata."""

    article: Article
    relevance_score: int
    impact_level: ImpactLevel
    impact_type: ImpactType
    score_breakdown: ScoreBreakdown

    # Convenience pass-throughs
    @property
    def title(self) -> str:
        return self.article.title

    @property
    def source(self) -> str:
        return self.article.source

    @property
    def url(self) -> str:
        return self.article.url

    @property
    def published_date(self) -> Optional[datetime]:
        return self.article.published_date


@dataclass
class SummarizedArticle:
    """Final output: scored article plus a sales-impact summary."""

    scored: ScoredArticle
    summary: str

    # Convenience pass-throughs
    @property
    def title(self) -> str:
        return self.scored.title

    @property
    def source(self) -> str:
        return self.scored.source

    @property
    def url(self) -> str:
        return self.scored.url

    @property
    def published_date(self) -> Optional[datetime]:
        return self.scored.published_date

    @property
    def relevance_score(self) -> int:
        return self.scored.relevance_score

    @property
    def impact_level(self) -> ImpactLevel:
        return self.scored.impact_level

    @property
    def impact_type(self) -> ImpactType:
        return self.scored.impact_type

    def to_dict(self) -> dict:
        pub = self.scored.published_date
        return {
            "title": self.title,
            "source": self.scored.source,
            "url": self.scored.url,
            "published_date": pub.strftime("%Y-%m-%d") if pub else None,
            "relevance_score": self.relevance_score,
            "impact_level": self.impact_level,
            "impact_type": self.impact_type,
            "summary": self.summary,
        }
