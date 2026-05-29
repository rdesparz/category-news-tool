"""
FastAPI web application for the Category News Intelligence Tool.

Run:
    uvicorn web.app:app --host 0.0.0.0 --port 8080 --reload
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import anthropic
import yaml
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fetcher.news_fetcher import fetch_news_for_category
from src.keywords.category_keywords import get_keywords
from src.models.article import Article
from src.registry.categories import list_configured_categories
from src.report.report_generator import build_report_data, save_report
from src.scorer.relevance_scorer import score_articles
from src.summarizer.summarizer import summarize_articles

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Category News Intelligence",
    description="Fetch, score, summarize, and report on category news for Amazon product categories.",
    version="1.0.0",
)

# Serve static files (CSS, JS)
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

_CONFIG_PATH = str(Path(__file__).parent.parent / "config.yaml")
_CATEGORY_DIR = str(Path(__file__).parent.parent / "data" / "category_configs")


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured on the server.")
    return anthropic.Anthropic(api_key=api_key)


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/api/categories")
def api_list_categories():
    """List all registered categories."""
    return {"categories": list_configured_categories(_CATEGORY_DIR)}


@app.post("/api/categories")
def api_add_category(name: str = Query(...), keywords: str = Query(...)):
    """Register a new category with comma-separated keywords."""
    terms = [t.strip() for t in keywords.split(",") if t.strip()]
    if not terms:
        raise HTTPException(status_code=400, detail="No keywords provided.")

    slug = name.lower().replace(" ", "_")
    path = Path(_CATEGORY_DIR) / f"{slug}.yaml"
    if path.exists():
        raise HTTPException(status_code=409, detail=f"Category '{name}' already exists.")

    data = {
        "category": name,
        "keywords": {
            "primary": terms[:5],
            "brands": [t for t in terms if t[0].isupper()][:8],
            "supply_chain": [t for t in terms if any(w in t.lower() for w in ("recall", "shortage", "supply"))],
            "seasonal": [t for t in terms if any(w in t.lower() for w in ("season", "winter", "summer"))],
            "regulatory": [t for t in terms if any(w in t.lower() for w in ("regulation", "standard", "fda"))],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    return {"status": "created", "category": name, "path": str(path)}


@app.get("/api/report")
def api_generate_report(
    category: str = Query(..., description="Category name"),
    days: int = Query(7, description="Lookback period in days"),
    format: str = Query("json", description="Output format: json, markdown, html"),
):
    """Run the full pipeline and return the report."""
    config = _load_config()
    client = _get_client()

    fetcher_cfg = config.get("fetcher", {})
    scorer_cfg = config.get("scorer", {})
    summarizer_cfg = config.get("summarizer", {})
    anthropic_cfg = config.get("anthropic", {})
    model = anthropic_cfg.get("model", "claude-haiku-4-5")
    cache_dir = fetcher_cfg.get("cache_dir", ".cache")
    threshold = scorer_cfg.get("threshold", 30)

    # 1 — Fetch
    raw_dicts = fetch_news_for_category(category, config=config, days=days)
    raw_articles = [
        Article(
            title=d.get("title", ""),
            source=d.get("source", ""),
            url=d.get("url", ""),
            published_date=d.get("published_date"),
            snippet=d.get("snippet", ""),
        )
        for d in raw_dicts
    ]

    # 2 — Score
    keywords = get_keywords(category, config=config, anthropic_client=client)
    scored = score_articles(raw_articles, keywords, threshold=threshold)

    # 3 — Summarize
    summarized = summarize_articles(
        scored,
        category=category,
        client=client,
        model=model,
        max_output_tokens=summarizer_cfg.get("max_output_tokens", 1024),
        max_retries=summarizer_cfg.get("max_retries", 3),
        base_delay=summarizer_cfg.get("retry_base_delay", 1.0),
        cache_dir=cache_dir,
    )

    # 4 — Build report
    report_data = build_report_data(
        category=category,
        articles_analyzed=len(raw_articles),
        summarized=summarized,
        days=days,
        client=client,
        model=model,
        max_retries=summarizer_cfg.get("max_retries", 3),
        base_delay=summarizer_cfg.get("retry_base_delay", 1.0),
        sources_used=["Google News RSS", "NewsAPI", "GNews"],
    )

    # 5 — Save
    output_dir = config.get("report", {}).get("output_dir", "reports")
    report_path = save_report(report_data, fmt=format, output_dir=output_dir)

    if format == "json":
        return JSONResponse(content=report_data)
    elif format == "html":
        return HTMLResponse(content=report_path.read_text(encoding="utf-8"))
    else:
        return JSONResponse(content={
            "status": "complete",
            "report_path": str(report_path),
            "report_data": report_data,
        })


@app.get("/api/report/quick")
def api_quick_report(
    category: str = Query(..., description="Category name"),
    days: int = Query(7),
):
    """Run fetch + score only (no LLM summarization). Fast preview."""
    config = _load_config()

    raw_dicts = fetch_news_for_category(category, config=config, days=days)
    raw_articles = [
        Article(
            title=d.get("title", ""),
            source=d.get("source", ""),
            url=d.get("url", ""),
            published_date=d.get("published_date"),
            snippet=d.get("snippet", ""),
        )
        for d in raw_dicts
    ]

    try:
        client = _get_client()
    except HTTPException:
        client = None

    keywords = get_keywords(category, config=config, anthropic_client=client)
    scored = score_articles(raw_articles, keywords, threshold=config.get("scorer", {}).get("threshold", 30))

    return {
        "category": category,
        "articles_scanned": len(raw_articles),
        "articles_relevant": len(scored),
        "articles": [
            {
                "title": a.title,
                "source": a.source,
                "url": a.url,
                "published_date": a.published_date.isoformat() if a.published_date else None,
                "relevance_score": a.relevance_score,
                "impact_level": a.impact_level,
                "impact_type": a.impact_type,
            }
            for a in scored
        ],
    }


# ── Frontend ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    """Serve the single-page frontend."""
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
