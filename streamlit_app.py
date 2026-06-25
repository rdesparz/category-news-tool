"""
Streamlit dashboard for the Category News Intelligence Tool.

Run:
    streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime
from pathlib import Path

import anthropic
import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.fetcher.news_fetcher import fetch_news_for_category
from src.keywords.category_keywords import get_keywords
from src.models.article import Article
from src.registry.categories import list_configured_categories
from src.report.report_generator import build_report_data, save_report
from src.scorer.relevance_scorer import score_articles
from src.summarizer.summarizer import summarize_articles

CONFIG_PATH = "config.yaml"
CATEGORY_DIR = "data/category_configs"

st.set_page_config(
    page_title="Category News Intelligence",
    page_icon="📰",
    layout="wide",
)

# Analytics — use pixel tracking (JS doesn't work in Streamlit iframes)
st.markdown(
    '<img src="https://resparza.goatcounter.com/count?p=/visit&t=Category%20News%20Intelligence" '
    'style="position:absolute;top:-9999px;" alt="">',
    unsafe_allow_html=True,
)

# ── Amazon-style CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Amazon dark header bar */
[data-testid="stHeader"] {
    background-color: #232F3E;
}

/* Primary buttons → Amazon Orange */
.stButton > button[kind="primary"],
button[data-testid="stFormSubmitButton"] {
    background-color: #FF9900 !important;
    color: #111 !important;
    border: 1px solid #E88B00 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
.stButton > button[kind="primary"]:hover,
button[data-testid="stFormSubmitButton"]:hover {
    background-color: #E88B00 !important;
}

/* Full-width primary buttons (Generate Report) → big & featured */
.stButton > button[kind="primary"][data-testid="baseButton-primary"] {
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    padding: 0.85rem 1rem !important;
    box-shadow: 0 4px 14px rgba(255,153,0,0.4) !important;
    letter-spacing: 0.02em;
    border: 2px solid #E88B00 !important;
    transition: all 0.15s ease !important;
}
.stButton > button[kind="primary"][data-testid="baseButton-primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(255,153,0,0.5) !important;
}

/* Secondary buttons */
.stButton > button:not([kind="primary"]) {
    border-radius: 8px !important;
    border: 1px solid #D5D9D9 !important;
    background: #FFF !important;
}
.stButton > button:not([kind="primary"]):hover {
    background: #F7FAFA !important;
}

/* Tabs → Amazon style */
.stTabs [data-baseweb="tab-list"] {
    gap: 0px;
    border-bottom: 2px solid #D5D9D9;
}
.stTabs [data-baseweb="tab"] {
    padding: 10px 20px;
    font-weight: 600;
    color: #565959;
}
.stTabs [aria-selected="true"] {
    color: #C45500 !important;
    border-bottom: 3px solid #FF9900 !important;
}

/* Sidebar toggle arrows — white in both states */
[data-testid="stSidebar"] button svg,
[data-testid="collapsedControl"] svg,
[data-testid="collapsedControl"] path,
[data-testid="collapsedControl"] button,
[data-testid="collapsedControl"] button svg,
[data-testid="collapsedControl"] button svg path,
[data-testid="stSidebar"] [data-testid="collapsedControl"] svg,
button[kind="header"] svg,
[data-testid="baseButton-header"] svg,
[data-testid="baseButton-header"] svg path {
    fill: #FFFFFF !important;
    stroke: #FFFFFF !important;
    color: #FFFFFF !important;
}
[data-testid="collapsedControl"],
[data-testid="collapsedControl"] button,
[data-testid="baseButton-header"] {
    background-color: #232F3E !important;
    border-radius: 0 0 8px 0;
}

/* Mobile & desktop: hamburger icon for collapsed sidebar */
[data-testid="collapsedControl"] button svg,
[data-testid="collapsedControl"] button svg path {
    display: none !important;
}
[data-testid="collapsedControl"] button {
    background-color: #232F3E !important;
    border: none !important;
    width: 40px !important;
    height: 40px !important;
    position: relative !important;
}
[data-testid="collapsedControl"] button::after {
    content: "☰" !important;
    font-size: 22px !important;
    color: #FFFFFF !important;
    position: absolute !important;
    top: 50% !important;
    left: 50% !important;
    transform: translate(-50%, -50%) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #232F3E;
}
[data-testid="stSidebar"] * {
    color: #F2F2F2 !important;
}
[data-testid="stSidebar"] .stSelectbox > label,
[data-testid="stSidebar"] .stMultiSelect > label,
[data-testid="stSidebar"] .stRadio > label {
    color: #FF9900 !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    font-size: 0.75rem !important;
    letter-spacing: 0.03em;
}
/* Radio option text — match filter checkbox size, keep on one line */
[data-testid="stSidebar"] .stRadio [role="radiogroup"] label p {
    white-space: nowrap !important;
}
/* Tighten checkbox spacing to match radio option spacing */
[data-testid="stSidebar"] .stCheckbox {
    margin-bottom: -0.5rem !important;
}
/* Multiselect dropdown arrow — navy for visibility */
[data-testid="stSidebar"] [data-testid="stMultiSelect"] [data-baseweb="select"] > div > div:last-child svg {
    fill: #232F3E !important;
    stroke: #232F3E !important;
    color: #232F3E !important;
    width: 20px !important;
    height: 20px !important;
}
/* Multiselect tag X buttons — smaller and white */
[data-testid="stSidebar"] [data-testid="stMultiSelect"] [data-baseweb="tag"] svg {
    fill: #FFFFFF !important;
    stroke: #FFFFFF !important;
    color: #FFFFFF !important;
    width: 12px !important;
    height: 12px !important;
}
[data-testid="stSidebar"] [data-testid="stMultiSelect"] [data-baseweb="tag"] [role="presentation"] {
    width: 16px !important;
    height: 16px !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #D5D9D9;
    border-radius: 8px;
    padding: 12px 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
[data-testid="stMetricValue"] {
    color: #C45500 !important;
    font-weight: 700 !important;
}

/* Expanders */
.streamlit-expanderHeader {
    font-weight: 600;
    color: #0F1111;
    border: 1px solid #D5D9D9;
    border-radius: 8px;
}

/* Info/success/warning boxes */
.stAlert > div[data-baseweb="notification"] {
    border-radius: 8px !important;
}

/* Links → Amazon blue */
a {
    color: #007185 !important;
}
a:hover {
    color: #C7511F !important;
    text-decoration: underline !important;
}

/* Panel/container borders */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"] > [data-testid="stMarkdown"]) {
    border: 1px solid #D5D9D9 !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_anthropic_client() -> anthropic.Anthropic | None:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)


def impact_color(level: str) -> str:
    return {"High": "🟠", "Medium": "🔶", "Low": "🟡"}.get(level, "⚪")


# Categories most likely to surface broad, high-impact breaking news.
_BREAKING_NEWS_CATEGORIES = [
    "Electronics", "Wireless", "PC", "Mobile Electronics",
    "Apparel", "Grocery", "Home", "Toys",
]
# Terms that signal a high-priority breaking story right now (Prime week + pricing).
_BREAKING_BOOST_TERMS = {
    "prime day": 25, "prime week": 25, "prime big deal": 20,
    "price increase": 20, "raising prices": 20, "price hike": 20,
    "tariff": 15, "raise prices": 20, "hikes price": 20,
}


@st.cache_data(ttl=900, show_spinner=False)
def scan_all_categories() -> list[dict]:
    """
    Scan across major categories once and return ALL scored articles as dicts.

    This is the shared data source for breaking news and all homepage widgets,
    so the page only fetches/scores once. Cached for 15 minutes; underlying
    article fetch refreshes every 2 hours (see config fetcher.cache_ttl_hours).
    """
    config = load_config()
    articles: list[dict] = []
    seen_urls: set[str] = set()

    for category in _BREAKING_NEWS_CATEGORIES:
        try:
            raw_dicts = fetch_news_for_category(category, config=config, days=3, use_cache=True)
        except Exception:
            continue

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

        keywords = get_keywords(category, config=config)
        threshold = config.get("scorer", {}).get("threshold", 30)
        scored = score_articles(raw_articles, keywords, threshold=threshold)

        for art in scored:
            if art.url in seen_urls:
                continue
            seen_urls.add(art.url)

            text = f"{art.title} {art.article.snippet}".lower()
            boost = sum(pts for term, pts in _BREAKING_BOOST_TERMS.items() if term in text)

            articles.append({
                "title": art.title,
                "source": art.source,
                "url": art.url,
                "category": category,
                "impact_type": art.impact_type,
                "impact_level": art.impact_level,
                "relevance_score": art.relevance_score,
                "effective_score": art.relevance_score + boost,
                "published_date": art.published_date.strftime("%Y-%m-%d") if art.published_date else None,
                "published_ts": art.published_date.timestamp() if art.published_date else 0,
                "is_prime": any(t in text for t in ("prime day", "prime week", "prime big deal")),
            })

    return articles


# Authoritative AND freely-accessible sources — preferred for the featured slot.
# (Paywalled/bot-blocking sources are excluded here and penalized below so the
#  featured link actually opens for users.)
_TOP_TIER_SOURCES = {
    "abc news", "cnbc", "techcrunch", "the verge", "axios", "bbc", "cnn",
    "associated press", "ap", "engadget", "the register", "9to5mac",
    "appleinsider", "macrumors", "macdailynews", "cbc", "mashable",
    "the guardian", "quartz", "qz.com", "zdnet", "ars technica",
}
# Sources that frequently serve paywalls or bot-check pages — never feature these
# (the link often won't load for the user).
_PAYWALL_SOURCES = {
    "barron's", "barrons", "bloomberg", "the wall street journal", "wsj",
    "the new york times", "nyt", "financial times", "ft.com", "the information",
    "business insider", "forbes", "the economist", "fortune", "reuters",
}
# Headline patterns that signal a low-value "deals roundup" rather than real news.
_DEALS_ROUNDUP_TERMS = (
    "best deals", "best sales", "deals rival", "save on", "discount",
    "get the", "% off", "last chance", "best prime day", "top deals",
    "deals we found", "shop the", "before apple raises", "prime day deals",
    "deals even better", "deals include", "$ off", "off for prime",
    "off the", "discounted", "grab the", "snag the", "deal of the",
)
# Strong "this is real news reporting" signals for the featured slot.
_HARD_NEWS_TERMS = (
    "price hike", "price increase", "raises price", "hikes price",
    "raises prices", "go into effect", "price hikes", "more expensive",
    "announces price", "shares slide", "stock drops", "stock falls",
    "passes", "costs to consumers", "amid", "due to",
)


def _featured_quality(article: dict) -> int:
    """
    Score how well-suited an article is for the FEATURED hero slot.

    Strongly rewards authoritative sources + real news reporting;
    heavily penalizes 'deals roundup' headlines so they never feature.
    """
    title = article["title"].lower()
    source = article["source"].lower().strip()
    q = article["effective_score"]

    # Heavy penalty for paywalled/bot-blocking sources — link often won't open
    if any(src in source for src in _PAYWALL_SOURCES):
        q -= 150

    # Top-tier (free + reputable) source boost
    if any(src in source for src in _TOP_TIER_SOURCES):
        q += 60

    # Substantive news-reporting language
    if any(w in title for w in _HARD_NEWS_TERMS):
        q += 30

    # Heavy penalty for deals-roundup framing — these should never be featured
    if any(term in title for term in _DEALS_ROUNDUP_TERMS):
        q -= 100

    return q


def _top_breaking(articles: list[dict], limit: int = 3) -> list[dict]:
    """Pick the best-quality story for the hero slot, then fill with category diversity."""
    if not articles:
        return []

    # Featured story: highest featured-quality article overall (substantive source + news language).
    featured = max(articles, key=_featured_quality)
    picked: list[dict] = [featured]
    used: set[str] = {featured["category"]}
    picked_urls: set[str] = {featured["url"]}

    # Fill remaining slots by effective score, preferring new categories.
    ranked = sorted(articles, key=lambda a: a["effective_score"], reverse=True)
    for item in ranked:
        if len(picked) >= limit:
            break
        if item["url"] in picked_urls or item["category"] in used:
            continue
        picked.append(item)
        used.add(item["category"])
        picked_urls.add(item["url"])

    # Backfill if still short (fewer distinct categories than limit).
    if len(picked) < limit:
        for item in ranked:
            if len(picked) >= limit:
                break
            if item["url"] not in picked_urls:
                picked.append(item)
                picked_urls.add(item["url"])

    return picked[:limit]


def render_breaking_news() -> None:
    """Render the curated breaking news banner + live widgets at the top of the Run tab."""
    with st.spinner("Loading breaking news…"):
        try:
            articles = scan_all_categories()
        except Exception:
            articles = []

    if not articles:
        return

    stories = _top_breaking(articles, limit=3)

    # ── Breaking News banner ──────────────────────────────────────────────
    st.markdown(
        "<div style='background:#232F3E; padding:10px 16px; border-radius:8px 8px 0 0; margin-bottom:0;'>"
        "<span style='color:#FF9900; font-weight:700; font-size:1rem;'>🔴 BREAKING NEWS</span>"
        "<span style='color:#D5D9D9; font-size:0.8rem; margin-left:10px;'>Top stories across all categories · last 3 days</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Pick the featured story: highest-scoring (already first after diversity sort).
    featured = stories[0] if stories else None
    rest = stories[1:] if len(stories) > 1 else []

    if featured:
        prime_badge = "🔥 PRIME " if featured.get("is_prime") else ""
        badge = impact_color(featured["impact_level"])
        st.markdown(
            f"""<div style='border:3px solid #FF9900; border-radius:0 0 8px 8px; background:#FFF8EE;
                 padding:16px 20px; margin-bottom:14px; box-shadow:0 2px 8px rgba(255,153,0,0.25);'>
                <span style='background:#E01E2B; color:#FFF; font-size:0.72rem; font-weight:800;
                      padding:3px 10px; border-radius:10px; letter-spacing:0.04em;'>⭐ FEATURED</span>
                <span style='background:#FF9900; color:#111; font-size:0.72rem; font-weight:700;
                      padding:3px 10px; border-radius:10px; margin-left:6px;'>{prime_badge}{featured['category']}</span>
                <div style='font-size:1.15rem; font-weight:700; margin-top:10px; line-height:1.35;'>
                    {badge} <a href='{featured['url']}' style='color:#232F3E !important; text-decoration:none;'>{featured['title']}</a>
                </div>
                <div style='font-size:0.82rem; color:#565959; margin-top:6px;'>
                    {featured['source']} · {featured.get('published_date') or 'recent'} ·
                    {featured['impact_type']} · Score {featured['relevance_score']}/100
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

    if rest:
        cols = st.columns(len(rest))
        for col, story in zip(cols, rest):
            with col:
                with st.container(border=True):
                    prime_badge = "🔥 PRIME " if story.get("is_prime") else ""
                    badge = impact_color(story["impact_level"])
                    st.markdown(
                        f"<span style='background:#FF9900; color:#111; font-size:0.7rem; font-weight:700; "
                        f"padding:2px 8px; border-radius:10px;'>{prime_badge}{story['category']}</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**{badge} [{story['title']}]({story['url']})**")
                    st.caption(
                        f"{story['source']} · {story.get('published_date') or 'recent'} · "
                        f"{story['impact_type']} · Score {story['relevance_score']}/100"
                    )

    # ── Live widgets ──────────────────────────────────────────────────────
    _render_widgets(articles)
    st.divider()


# Prime Day 2026 event window (multi-day event)
_PRIME_DAY_START = date(2026, 6, 23)
_PRIME_DAY_END = date(2026, 6, 26)


def _render_widgets(articles: list[dict]) -> None:
    """Render the five live homepage widgets from the shared article scan."""
    from collections import Counter

    # Row 1: Prime Day countdown + Trending impact types
    c1, c2 = st.columns(2)

    with c1:
        with st.container(border=True):
            st.markdown("##### 📈 Prime Day Tracker")
            today = date.today()
            prime_count = sum(1 for a in articles if a["is_prime"])
            window = f"{_PRIME_DAY_START:%b %d}–{_PRIME_DAY_END:%b %d, %Y}"
            if today < _PRIME_DAY_START:
                st.metric("Days until Prime Day", (_PRIME_DAY_START - today).days, help=f"Prime Day: {window}")
            elif _PRIME_DAY_START <= today <= _PRIME_DAY_END:
                days_left = (_PRIME_DAY_END - today).days
                st.metric("🎉 Prime Day is LIVE", f"{days_left} day{'s' if days_left != 1 else ''} left", help=window)
            else:
                st.metric("Days since Prime Day", (today - _PRIME_DAY_END).days)
            st.caption(f"🔥 {prime_count} Prime-related articles in the last 3 days")

    with c2:
        with st.container(border=True):
            st.markdown("##### 🔥 Trending Impact Types")
            type_counts = Counter(a["impact_type"] for a in articles)
            for itype, count in type_counts.most_common(4):
                st.markdown(f"**{itype}** — {count} stories")

    # Row 2: Category Heat Index + Category Activity Meter
    c3, c4 = st.columns(2)

    with c3:
        with st.container(border=True):
            st.markdown("##### 🌡️ Category Heat Index")
            st.caption("Top categories by high-impact stories")
            heat = Counter(
                a["category"] for a in articles if a["impact_level"] == "High"
            )
            if not heat:
                heat = Counter(a["category"] for a in articles)
            for cat, count in heat.most_common(5):
                st.markdown(f"🔴 **{cat}** — {count}")

    with c4:
        with st.container(border=True):
            st.markdown("##### 📊 Category Activity")
            st.caption("Article volume by category")
            activity = Counter(a["category"] for a in articles)
            if activity:
                max_count = max(activity.values())
                for cat, count in activity.most_common(6):
                    pct = int((count / max_count) * 100)
                    bar = "█" * max(1, int(pct / 10))
                    st.markdown(
                        f"<span style='font-size:0.8rem;'>{cat}</span> "
                        f"<span style='color:#FF9900;'>{bar}</span> "
                        f"<span style='font-size:0.75rem; color:#565959;'>{count}</span>",
                        unsafe_allow_html=True,
                    )

    # Row 3: Latest Headlines Ticker
    with st.container(border=True):
        st.markdown("##### 🕐 Latest Headlines")
        recent = sorted(articles, key=lambda a: a["published_ts"], reverse=True)[:5]
        for a in recent:
            st.markdown(
                f"<span style='font-size:0.85rem;'>🔹 [{a['title']}]({a['url']}) "
                f"<span style='color:#565959;'>· {a['category']} · {a.get('published_date') or 'recent'}</span></span>",
                unsafe_allow_html=True,
            )


def _report_to_excel(file_path: Path) -> bytes | None:
    """Convert a report JSON file to an Excel workbook. Returns bytes or None."""
    import io
    try:
        import openpyxl
    except ImportError:
        return None

    # Only JSON reports can be converted
    if file_path.suffix == ".json":
        import json
        with open(file_path) as f:
            data = json.load(f)
    elif file_path.suffix == ".md":
        # For markdown, try to find a sibling .json
        json_path = file_path.with_suffix(".json")
        if json_path.exists():
            import json
            with open(json_path) as f:
                data = json.load(f)
        else:
            return None
    else:
        return None

    wb = openpyxl.Workbook()

    # Sheet 1: Summary
    ws = wb.active
    ws.title = "Summary"
    ws.append(["Category", data.get("category", "")])
    ws.append(["Date Range", f"{data.get('date_start', '')} – {data.get('date_end', '')}"])
    ws.append(["Articles Scanned", data.get("articles_analyzed", 0)])
    ws.append(["Articles Relevant", data.get("relevant_articles", 0)])
    ws.append(["Generated", data.get("generated_at", "")])
    ws.append([])
    ws.append(["Executive Summary"])
    for bullet in data.get("executive_summary", []):
        ws.append(["", bullet])
    ws.append([])
    ws.append(["Recommended Actions"])
    for action in data.get("recommended_actions", []):
        ws.append(["", action])

    # Sheet 2: Articles
    ws2 = wb.create_sheet("Articles")
    ws2.append(["Title", "Source", "Date", "Score", "Impact Level", "Impact Type", "Summary", "URL"])
    for section in data.get("sections", []):
        for art in section.get("articles", []):
            ws2.append([
                art.get("title", ""),
                art.get("source", ""),
                art.get("published_date", ""),
                art.get("relevance_score", ""),
                art.get("impact_level", ""),
                art.get("impact_type", section.get("impact_type", "")),
                art.get("summary", ""),
                art.get("url", ""),
            ])

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


_PRIME_DAY_TERMS = {"prime day", "prime week", "prime deal", "prime deals", "prime big deal", "prime big deals"}


def _is_prime_day_article(article) -> bool:
    """Check if an article mentions Prime Day / Prime Week."""
    text = f"{article.title} {getattr(article, 'snippet', '')}".lower()
    return any(term in text for term in _PRIME_DAY_TERMS)


def run_pipeline(category: str, days: int, mode: str, status_box, no_cache: bool = False, prime_day_filter: bool = False) -> dict | None:
    """Run the full pipeline; mode is 'full' or 'quick'."""
    config = load_config()

    # 1 — Fetch
    status_box.info(f"📡 Fetching news for **{category}** ({days}-day lookback)…")
    raw_dicts = fetch_news_for_category(category, config=config, days=days, use_cache=not no_cache)
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

    # Apply Prime Day filter if enabled
    if prime_day_filter:
        raw_articles = [a for a in raw_articles if _is_prime_day_article(a)]
        status_box.info(f"🔥 Prime Day filter: {len(raw_articles)} articles mention Prime Day")

    # 2 — Score
    status_box.info(f"🎯 Scoring {len(raw_articles)} articles for relevance…")
    client = get_anthropic_client()
    using_templates = client is None and mode == "full"

    keywords = get_keywords(category, config=config, anthropic_client=client)
    threshold = config.get("scorer", {}).get("threshold", 30)
    scored = score_articles(raw_articles, keywords, threshold=threshold)

    if mode == "quick":
        status_box.success(f"✅ Found {len(scored)} relevant articles.")
        return {
            "mode": "quick",
            "category": category,
            "articles_scanned": len(raw_articles),
            "articles_relevant": len(scored),
            "scored": scored,
        }

    # 3 — Summarize
    if using_templates:
        status_box.info(f"📝 Building summaries for {len(scored)} articles…")
    else:
        status_box.info(f"🤖 Summarizing {len(scored)} relevant articles via Claude…")
    summarizer_cfg = config.get("summarizer", {})
    model = config.get("anthropic", {}).get("model", "claude-haiku-4-5")
    summarized = summarize_articles(
        scored,
        category=category,
        client=client,
        model=model,
        max_output_tokens=summarizer_cfg.get("max_output_tokens", 1024),
        max_retries=summarizer_cfg.get("max_retries", 3),
        base_delay=summarizer_cfg.get("retry_base_delay", 1.0),
        cache_dir=config.get("fetcher", {}).get("cache_dir", ".cache"),
    )

    # 4 — Report
    if using_templates:
        status_box.info("📋 Building template executive summary and playbook actions…")
    else:
        status_box.info("📝 Generating executive summary and recommended actions via Claude…")
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

    # 5 — Save to disk (markdown + json for Excel export)
    output_dir = config.get("report", {}).get("output_dir", "reports")
    saved_path = save_report(report_data, fmt="markdown", output_dir=output_dir)
    save_report(report_data, fmt="json", output_dir=output_dir)

    if using_templates:
        status_box.success(f"✅ Report complete. Also saved to `{saved_path}`")
    else:
        status_box.success(f"✅ Report complete. Also saved to `{saved_path}`")
    return {"mode": "full", "report": report_data, "saved_path": str(saved_path), "templates": using_templates}


# ── UI Components ─────────────────────────────────────────────────────────────

def render_sidebar() -> tuple[list[str], int, str, bool, bool]:
    logo = _logo_b64()
    st.sidebar.markdown(f"""
    <div style="text-align:center; padding:10px 0 15px;">
        <img src="data:image/svg+xml;base64,{logo}" style="height:36px; border-radius:6px;" alt="Amazon"/><br>
        <span style="color:#FF9900; font-size:0.75rem; font-weight:600; letter-spacing:0.05em; margin-top:6px; display:inline-block;">CATEGORY INTEL</span>
    </div>
    """, unsafe_allow_html=True)

    cats = list_configured_categories(CATEGORY_DIR)
    if not cats:
        st.sidebar.error("No categories registered.")
        st.sidebar.markdown("Register one in the **Manage Categories** tab.")

    selected = st.sidebar.multiselect(
        "Categories",
        options=cats,
        default=cats[:1] if cats else [],
        help="Pick one or more categories to analyze",
    )

    days = st.sidebar.select_slider("Lookback period (days)", options=[3, 7, 14, 30], value=7)

    mode = st.sidebar.radio(
        "Mode",
        options=["quick", "full"],
        format_func=lambda x: "⚡ Rapid Scan (recommended)" if x == "quick" else "📊 Categorized Report",
        help="Rapid Scan gives instant scored results. Categorized Report groups articles by impact type with summaries.",
    )

    st.sidebar.markdown("<span style='color:#FF9900; font-weight:600; font-size:0.75rem; letter-spacing:0.03em;'>FILTERS</span>", unsafe_allow_html=True)
    prime_day_filter = st.sidebar.checkbox("🔥 Prime Day only", help="Show only articles mentioning Prime Day, Prime Week, or Prime deals")
    no_cache = st.sidebar.checkbox("🔄 Force refresh", help="Get the latest news instead of cached results (cache is 6 hours)")

    st.sidebar.divider()

    return selected, days, mode, no_cache, prime_day_filter


def render_full_report(report: dict) -> None:
    # Header card
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.subheader(f"📰 {report['category']}")
        st.caption(f"📅 {report['date_start']} – {report['date_end']}")
    with col2:
        st.metric("Articles Scanned", report["articles_analyzed"])
    with col3:
        st.metric("Relevant", report["relevant_articles"])

    st.divider()
    st.markdown("### 🗂️ Detailed Findings")

    # Tabs by impact type
    sections = report.get("sections", [])
    if not sections:
        st.info("No relevant articles found.")
        return

    tab_labels = [f"{s['impact_type']} ({len(s['articles'])})" for s in sections]
    tabs = st.tabs(tab_labels)

    for tab, section in zip(tabs, sections):
        with tab:
            for art in section["articles"]:
                badge = impact_color(art["impact_level"])
                with st.container(border=True):
                    st.markdown(f"#### {badge} [{art['title']}]({art['url']})")
                    cols = st.columns([2, 2, 1, 1])
                    cols[0].caption(f"**Source:** {art['source']}")
                    cols[1].caption(f"**Date:** {art.get('published_date') or 'unknown'}")
                    cols[2].caption(f"**Score:** {art['relevance_score']}/100")
                    cols[3].caption(f"**Level:** {art['impact_level']}")
                    st.markdown(art["summary"])

    # Executive summary & actions below findings
    st.divider()
    with st.container(border=True):
        st.markdown("### 📋 Executive Summary")
        for bullet in report.get("executive_summary", []):
            st.markdown(f"- {bullet}")

    with st.container(border=True):
        st.markdown("### 💡 Recommended Actions")
        for action in report.get("recommended_actions", []):
            st.markdown(f"- {action}")

    # Footer
    st.divider()
    with st.expander("Report Metadata"):
        st.write(f"**Generated:** {report.get('generated_at')}")
        st.write(f"**Sources:** {', '.join(report.get('sources_used', []))}")
        st.caption(report.get("disclaimer", ""))


def render_quick_report(result: dict) -> None:
    st.subheader(f"⚡ Quick Scan: {result['category']}")
    col1, col2 = st.columns(2)
    col1.metric("Articles Scanned", result["articles_scanned"])
    col2.metric("Relevant", result["articles_relevant"])

    st.divider()
    scored = result["scored"]
    if not scored:
        st.info("No relevant articles found.")
        return

    # Group by impact level
    high = [a for a in scored if a.impact_level == "High"]
    med = [a for a in scored if a.impact_level == "Medium"]
    low = [a for a in scored if a.impact_level == "Low"]

    for label, group, color in [
        ("🟠 High Impact", high, "red"),
        ("🔶 Medium Impact", med, "orange"),
        ("🟡 Low Impact", low, "green"),
    ]:
        if not group:
            continue
        with st.expander(f"{label} ({len(group)} articles)", expanded=(color == "red")):
            for a in group:
                pub = a.published_date.strftime("%Y-%m-%d") if a.published_date else "unknown"
                st.markdown(
                    f"**[{a.title}]({a.url})**  \n"
                    f"*{a.source} · {pub} · Score: {a.relevance_score}/100 · "
                    f"{a.impact_type}*"
                )

    st.caption("⚡ Rapid Scan shows scored articles by impact level. Switch to 'Categorized Report' for articles grouped by impact type with summaries and recommended actions.")


_KEYWORD_GROUPS = ["primary", "brands", "supply_chain", "seasonal", "regulatory"]


def _render_edit_form(cat: str, slug: str, yaml_path: Path, data: dict, kw: dict) -> None:
    """Render an editable form for an existing category."""
    with st.form(f"edit_form_{slug}"):
        new_name = st.text_input("Category name", value=cat, key=f"name_{slug}")

        edited_groups: dict[str, str] = {}
        for group in _KEYWORD_GROUPS:
            current = ", ".join(kw.get(group, []))
            edited_groups[group] = st.text_area(
                group.replace("_", " ").title(),
                value=current,
                key=f"{group}_{slug}",
                height=70,
                help=f"Comma-separated keywords for the '{group}' group",
            )

        col1, col2 = st.columns([1, 4])
        save = col1.form_submit_button("💾 Save Changes", type="primary")
        if save:
            new_name_clean = new_name.strip()
            if not new_name_clean:
                st.error("Category name cannot be empty.")
                return

            new_slug = new_name_clean.lower().replace(" ", "_")
            new_path = Path(CATEGORY_DIR) / f"{new_slug}.yaml"

            # Block rename to a slug that's already in use by a different category
            if new_slug != slug and new_path.exists():
                st.error(f"A category named '{new_name_clean}' already exists.")
                return

            new_kw = {
                group: [t.strip() for t in text.split(",") if t.strip()]
                for group, text in edited_groups.items()
            }
            new_data = {"category": new_name_clean, "keywords": new_kw}

            try:
                # Write to new path; if renaming, remove the old file
                new_path.parent.mkdir(parents=True, exist_ok=True)
                with open(new_path, "w") as f:
                    yaml.dump(new_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
                if new_slug != slug:
                    yaml_path.unlink()
                st.success(f"✓ Saved changes to '{new_name_clean}'")
                st.cache_data.clear()
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to save: {exc}")


def render_manage_categories_tab() -> None:
    st.subheader("Registered Categories")

    cats = list_configured_categories(CATEGORY_DIR)
    if cats:
        for cat in cats:
            slug = cat.lower().replace(" ", "_")
            yaml_path = Path(CATEGORY_DIR) / f"{slug}.yaml"
            if not yaml_path.exists():
                continue

            with open(yaml_path) as f:
                data = yaml.safe_load(f)
            kw = data.get("keywords", {})

            with st.expander(cat):
                _render_edit_form(cat, slug, yaml_path, data, kw)

                st.divider()
                confirm_key = f"confirm_delete_{slug}"
                if st.session_state.get(confirm_key):
                    st.warning(f"⚠ Permanently delete **{cat}** and its config file?")
                    col1, col2, _ = st.columns([1, 1, 3])
                    if col1.button("Yes, delete", key=f"yes_{slug}", type="primary"):
                        try:
                            yaml_path.unlink()
                            st.session_state[confirm_key] = False
                            st.success(f"✓ Deleted '{cat}'")
                            st.cache_data.clear()
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Failed to delete: {exc}")
                    if col2.button("Cancel", key=f"no_{slug}"):
                        st.session_state[confirm_key] = False
                        st.rerun()
                else:
                    if st.button("🗑️ Delete category", key=f"del_{slug}"):
                        st.session_state[confirm_key] = True
                        st.rerun()
    else:
        st.info("No categories registered yet.")

    st.divider()
    st.subheader("➕ Add New Category")
    with st.form("add_category_form"):
        name = st.text_input("Category name", placeholder="e.g. Pet Food")
        keywords = st.text_area(
            "Keywords (comma-separated)",
            placeholder="dog food, cat food, Purina, Blue Buffalo, pet food recall",
            help="Include product terms, brand names, supply-chain terms, regulatory bodies.",
        )
        submitted = st.form_submit_button("Add Category", type="primary")
        if submitted:
            if not name or not keywords:
                st.error("Both name and keywords are required.")
            else:
                terms = [t.strip() for t in keywords.split(",") if t.strip()]
                slug = name.lower().replace(" ", "_")
                path = Path(CATEGORY_DIR) / f"{slug}.yaml"
                if path.exists():
                    st.error(f"Category '{name}' already exists.")
                else:
                    data = {
                        "category": name,
                        "keywords": {
                            "primary": terms[:5],
                            "brands": [t for t in terms if t[0:1].isupper()][:8],
                            "supply_chain": [t for t in terms if any(w in t.lower() for w in ("recall", "shortage", "supply"))],
                            "seasonal": [t for t in terms if any(w in t.lower() for w in ("season", "winter", "summer"))],
                            "regulatory": [t for t in terms if any(w in t.lower() for w in ("regulation", "standard", "fda"))],
                        },
                    }
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with open(path, "w") as f:
                        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
                    st.success(f"✓ Added '{name}'. Refresh the sidebar to use it.")
                    st.cache_data.clear()


# ── Main ──────────────────────────────────────────────────────────────────────

def _logo_b64() -> str:
    logo_path = Path(__file__).parent / "web" / "static" / "amazon_logo_b64.txt"
    if logo_path.exists():
        return logo_path.read_text().strip()
    return ""


def main() -> None:
    logo = _logo_b64()
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:16px; margin-bottom:8px;">
        <img src="data:image/svg+xml;base64,{logo}" style="height:48px; border-radius:8px;" alt="Amazon"/>
        <div>
            <span style="font-size:1.5rem; font-weight:700; color:#232F3E;">Category News Intelligence</span><br>
            <span style="font-size:0.85rem; color:#565959;">Automated news monitoring for Amazon product categories</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")

    selected_categories, days, mode, no_cache, prime_day_filter = render_sidebar()

    tab_run, tab_manage, tab_history, tab_help = st.tabs(["🚀 Run Report", "⚙️ Manage Categories", "📂 Report History", "❓ How to Use"])

    # ── Run tab ──────────────────────────────────────────────────────────────
    with tab_run:
        run_btn = st.button("🚀 Generate Report", type="primary", use_container_width=True)

        # ── Report results appear here, directly under the button ──────────────
        if run_btn and not selected_categories:
            st.warning("👈 Select one or more categories from the sidebar first.")
        elif run_btn:
            for cat in selected_categories:
                st.markdown(f"## {cat}")
                status = st.empty()
                result = run_pipeline(cat, days, mode, status, no_cache=no_cache, prime_day_filter=prime_day_filter)
                status.empty()
                if result is None:
                    continue
                if result["mode"] == "full":
                    render_full_report(result["report"])
                else:
                    render_quick_report(result)
                st.divider()

        # ── Curated breaking news + live widgets (always shown) ────────────────
        render_breaking_news()

    # ── Manage tab ───────────────────────────────────────────────────────────
    with tab_manage:
        render_manage_categories_tab()

    # ── History tab ──────────────────────────────────────────────────────────
    with tab_history:
        st.subheader("Saved Reports")
        reports_dir = Path("reports")
        if not reports_dir.exists():
            st.info("No reports generated yet.")
        else:
            cat_dirs = sorted([d for d in reports_dir.iterdir() if d.is_dir()])
            if not cat_dirs:
                st.info("No reports generated yet.")
            for cat_dir in cat_dirs:
                with st.expander(f"📁 {cat_dir.name.replace('_', ' ').title()}"):
                    date_dirs = sorted([d for d in cat_dir.iterdir() if d.is_dir()], reverse=True)
                    for date_dir in date_dirs[:10]:
                        files = sorted(date_dir.glob("report.*"))
                        for f in files:
                            cols = st.columns([2, 1, 1])
                            cols[0].markdown(f"**{date_dir.name}** · `{f.name}`")
                            cols[1].markdown(f"*{f.stat().st_size // 1024} KB*")
                            with cols[2]:
                                excel_data = _report_to_excel(f)
                                if excel_data:
                                    st.download_button(
                                        "Download Excel",
                                        data=excel_data,
                                        file_name=f"{cat_dir.name}_{date_dir.name}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        key=str(f),
                                    )
                                else:
                                    with open(f, "rb") as fh:
                                        st.download_button(
                                            "Download",
                                            data=fh.read(),
                                            file_name=f.name,
                                            key=str(f),
                                        )


    # ── How to Use tab ────────────────────────────────────────────────────────
    with tab_help:
        st.subheader("How to Use This Tool")
        st.markdown("""
**Category News Intelligence** automatically scans public news sources, scores articles
for their relevance to your Amazon product category, and delivers an intelligence brief
you can use in seller calls, business reviews, or category planning.

---

### Quick Start (2 minutes)

1. **Select a category** from the sidebar dropdown (e.g. "Vitamins & Supplements")
2. **Set the lookback period** — how many days of news to scan (7 is a good default)
3. **Choose a mode:**
   - **⚡ Rapid Scan (recommended)** — instant results showing scored articles grouped by impact level. Fast, no wait.
   - **📊 Categorized Report** — articles grouped by impact type (Supply Chain, Pricing, Demand, etc.) with summaries, executive brief, and recommended actions.
4. **Click "Generate Report"**

---

### Rapid Scan vs. Categorized Report

| | ⚡ Rapid Scan | 📊 Categorized Report |
|---|---|---|
| **Speed** | Instant | 10-30 seconds |
| **Articles** | Scored and grouped by impact level (High/Medium/Low) | Grouped by impact type (Supply Chain, Pricing, Demand, etc.) |
| **Summaries** | Headlines + scores | Article summaries with sales-impact framing |
| **Extras** | — | Executive Summary + Recommended Actions |
| **Best for** | Quick pulse check, mid-week scan | Weekly briefs, sharing with team, business reviews |

---

### Impact Levels

- 🟠 **High (70-100):** Likely to directly affect pricing, availability, or demand
- 🔶 **Medium (40-69):** Worth monitoring — may affect the category within weeks
- 🟡 **Low (30-39):** Background signal — no immediate action needed

### Impact Types

- **Supply Chain** — shortages, factory closures, logistics disruptions
- **Pricing** — tariffs, cost increases, price wars
- **Demand** — trend shifts, viral moments, new tech adoption
- **Regulatory** — recalls, new laws, safety mandates
- **Competitive** — brand earnings, bankruptcies, product launches
- **Seasonal** — weather events, holiday demand shifts

---

### Filters

- **🔥 Prime Day only** — show only articles mentioning Prime Day, Prime Week, or Prime deals
- **🔄 Force refresh** — get the latest news instead of cached results (cache is 6 hours)

---

### Managing Categories

Go to the **⚙️ Manage Categories** tab to:
- **View** existing categories and their keyword configurations
- **Edit** category names and keyword groups (primary terms, brands, supply chain, seasonal, regulatory)
- **Add** new categories with comma-separated keywords
- **Delete** categories you no longer need

**Tips for good keywords:**
- **Primary:** Core product terms customers search for (e.g. "tires", "tire", "wheels")
- **Brands:** Major brands in the space (e.g. "Michelin", "Goodyear")
- **Supply Chain:** Sourcing/logistics terms (e.g. "rubber prices", "tire shortage")
- **Seasonal:** Seasonal triggers (e.g. "winter tires", "road trip season")
- **Regulatory:** Standards bodies and compliance (e.g. "NHTSA", "DOT regulations")

The more specific your keywords, the better the scoring. Generic terms produce more noise.

---

### Lookback Period

| Setting | Best for |
|---|---|
| **3 days** | Quick mid-week check for breaking news |
| **7 days** | Standard weekly review (recommended) |
| **14 days** | Catching up after time off |
| **30 days** | Monthly trend overview |

---

### Report History

All generated Categorized Reports are automatically saved. Go to **📂 Report History** to browse
and download past reports as Excel files.

---

### FAQ

**Q: Why are some articles scored low or missing?**
The scorer uses keyword matching against your category's configured keywords. If an article
doesn't mention any of your terms, it won't score high. Edit the category keywords to improve coverage.

**Q: Can I run multiple categories at once?**
Yes — select multiple categories in the sidebar dropdown before clicking Generate Report.

**Q: What does "Force refresh" do?**
Results are cached for 6 hours to avoid hitting news APIs repeatedly. Toggle Force refresh
to get the latest articles — useful when you just added keywords or want breaking news.
        """)


if __name__ == "__main__":
    main()
