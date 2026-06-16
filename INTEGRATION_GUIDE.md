# Category News Intelligence Tool — Integration Guide

Use this doc to recreate the Category News Intelligence tool within the MFN dashboard.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│  Streamlit Frontend (streamlit_app.py)               │
│  - Category selector, mode toggle, report viewer     │
└──────────────┬───────────────────────────────────────┘
               │ calls
┌──────────────▼───────────────────────────────────────┐
│  Pipeline Modules (src/)                             │
│  1. Fetcher   → Google News RSS, NewsAPI, GNews      │
│  2. Scorer    → Keyword-based relevance (0-100)      │
│  3. Summarizer→ Claude Haiku (batch, cached)         │
│  4. Reporter  → Executive summary + actions (Claude) │
└──────────────────────────────────────────────────────┘
               │ reads
┌──────────────▼───────────────────────────────────────┐
│  Category Configs (data/category_configs/*.yaml)     │
│  - Keywords: primary, brands, supply_chain,          │
│    seasonal, regulatory                              │
└──────────────────────────────────────────────────────┘
```

---

## Source Files

### Entry Points
| File | Purpose |
|------|---------|
| `streamlit_app.py` | Web dashboard (Streamlit) |
| `web/app.py` | FastAPI alternative (REST API + static frontend) |
| `main.py` | CLI entry point |

### Core Pipeline (`src/`)
| Module | Purpose |
|--------|---------|
| `src/models/article.py` | Data classes: Article, ScoredArticle, SummarizedArticle |
| `src/fetcher/news_fetcher.py` | Fetches news from 3 sources (Google RSS, NewsAPI, GNews), deduplicates, caches |
| `src/fetcher/rss_parser.py` | Google News RSS parser (no API key needed) |
| `src/keywords/category_keywords.py` | Loads keyword configs from YAML; falls back to Claude for unknown categories |
| `src/registry/categories.py` | Discovers available category YAML configs |
| `src/scorer/relevance_scorer.py` | Scores articles 0-100 across 7 dimensions |
| `src/summarizer/summarizer.py` | Batches articles to Claude for 2-3 sentence sales-impact summaries |
| `src/report/report_generator.py` | Generates executive summary, recommended actions, renders MD/HTML/JSON |

---

## Pipeline Flow

```
1. FETCH:    category → keywords → search queries → [Google RSS + NewsAPI + GNews]
             → deduplicate (fuzzy 85%) → date filter → cache (6hr TTL)

2. SCORE:    articles × keywords → 7 dimensions scored (additive, max 100)
             → filter below threshold (default 30)
             → assign impact_level (High/Medium/Low) + impact_type

3. SUMMARIZE: scored articles → batch (≤5 per API call) → Claude Haiku
              → 2-3 sentence sales-impact summaries (cached per URL)

4. REPORT:   summarized articles → Claude generates:
             - 3-5 executive summary bullets
             - 3-5 recommended actions
             → group by impact type → render as MD/HTML/JSON
```

---

## Scoring Rubric (7 dimensions, additive)

| Dimension | Max Points | What triggers it |
|-----------|-----------|-----------------|
| Direct product mention | +30 | Primary keywords match |
| Supply chain | +25 | Category supply_chain keywords OR universal supply signals |
| Pricing/tariff | +25 | Universal pricing signals (tariff, inflation, price hike, etc.) |
| Consumer demand | +20 | Universal demand signals (trending, viral, surge, etc.) |
| Competitor/brand | +15 | Category brand keywords OR universal competitor signals |
| Regulatory | +20 | Category regulatory keywords OR universal regulatory signals |
| Seasonal | +10 | Category seasonal keywords OR universal seasonal signals |

**Impact levels:** High (70-100), Medium (40-69), Low (30-39)

---

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `ANTHROPIC_API_KEY` | For LLM features | Claude API for summaries, exec summary, actions, keyword generation |
| `NEWS_API_KEY` | Optional | NewsAPI.org (increases article coverage) |
| `GNEWS_API_KEY` | Optional | GNews.io (increases article coverage) |

Without `ANTHROPIC_API_KEY`, the tool uses template-based summaries (still functional, just less detailed).

---

## Category Config Format (`data/category_configs/<slug>.yaml`)

```yaml
category: Tires
keywords:
  primary:
    - tires
    - tire
    - wheels
    - rims
    - all-season tires
  brands:
    - Michelin
    - Goodyear
    - Bridgestone
    - Continental
    - Pirelli
  supply_chain:
    - rubber prices
    - tire shortage
    - tire recall
    - rubber supply
  seasonal:
    - winter tires
    - snow tires
    - road trip season
    - summer tires
  regulatory:
    - NHTSA
    - DOT regulations
    - tire safety standard
    - TREAD Act
```

---

## Dependencies (`requirements.txt`)

```
anthropic>=0.40.0
requests>=2.31.0
feedparser>=6.0.11
thefuzz>=0.22.1
python-Levenshtein>=0.25.0
PyYAML>=6.0.1
python-dateutil>=2.9.0
click>=8.1.0
rich>=13.0.0
jinja2>=3.1.0
fastapi>=0.110.0
uvicorn>=0.29.0
python-multipart>=0.0.9
streamlit>=1.40.0
```

---

## Embedding in MFN Dashboard

### Option A: Embed as iframe
If MFN dashboard supports iframes, point at:
```
https://category-news-intelligence-tool.streamlit.app/
```

### Option B: Use as a backend service
Use the FastAPI app (`web/app.py`) as a microservice:
```
GET  /api/categories          → list all categories
POST /api/categories          → add a category (name, keywords)
GET  /api/report?category=X   → full pipeline, returns JSON report
GET  /api/report/quick?category=X → fetch+score only (no LLM, instant)
```

### Option C: Import the pipeline directly
If MFN dashboard is Python, import and call the pipeline:
```python
from src.fetcher.news_fetcher import fetch_news_for_category
from src.keywords.category_keywords import get_keywords
from src.models.article import Article
from src.scorer.relevance_scorer import score_articles
from src.summarizer.summarizer import summarize_articles
from src.report.report_generator import build_report_data

# 1. Fetch
raw_dicts = fetch_news_for_category("Tires", config=config, days=7)
articles = [Article(title=d["title"], source=d["source"], url=d["url"],
                    published_date=d["published_date"], snippet=d["snippet"])
            for d in raw_dicts]

# 2. Score
keywords = get_keywords("Tires", config=config, anthropic_client=client)
scored = score_articles(articles, keywords, threshold=30)

# 3. Summarize
summarized = summarize_articles(scored, category="Tires", client=client)

# 4. Report
report = build_report_data(category="Tires", articles_analyzed=len(articles),
                           summarized=summarized, days=7, client=client,
                           model="claude-haiku-4-5", max_retries=3, base_delay=1.0,
                           sources_used=["Google News RSS"])
```

---

## GitHub Repo

```
https://github.com/rdesparz/category-news-tool
```

Clone and run locally:
```bash
git clone https://github.com/rdesparz/category-news-tool.git
cd category-news-tool
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...
streamlit run streamlit_app.py --server.port 8501
```

---

## Docker

```bash
docker build -t category-news-tool .
docker run -p 8501:8501 -e ANTHROPIC_API_KEY=sk-... category-news-tool
```

---

## Current Categories (39 total)

Apparel, Art & Craft Supplies, Automotive, Baby Products, Beauty, BISS, Camera, Digital Product Accessories, Drugstore, Electronics, Furniture, Grocery, Guild, Home, Home Entertainment, Home Improvement, Jewelry, Kitchen, Lawn & Garden, Loose Stones, Luggage, Luxury Beauty, Major Appliances, Mobile Electronics, Musical Instruments, Office Products, Outdoors, PC, Pet Products, Shoes, Software, Sports, Sports Memorabilia, Tires, Tools, Toys, Video Games, Watch, Wireless
