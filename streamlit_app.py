"""
Streamlit dashboard for the Category News Intelligence Tool.

Run:
    streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
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

/* Sidebar collapse/expand arrow */
[data-testid="stSidebar"] button[aria-label="Collapse sidebar"],
[data-testid="collapsedControl"] button {
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] button[aria-label="Collapse sidebar"] svg,
[data-testid="collapsedControl"] svg {
    fill: #FFFFFF !important;
    stroke: #FFFFFF !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #232F3E;
}
[data-testid="stSidebar"] * {
    color: #F2F2F2 !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stRadio label {
    color: #FF9900 !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    font-size: 0.75rem !important;
    letter-spacing: 0.03em;
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


def run_pipeline(category: str, days: int, mode: str, status_box, no_cache: bool = False) -> dict | None:
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
        status_box.info(f"📝 Building template summaries for {len(scored)} articles (no API key set)…")
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

    # 5 — Save to disk too
    output_dir = config.get("report", {}).get("output_dir", "reports")
    saved_path = save_report(report_data, fmt="markdown", output_dir=output_dir)

    if using_templates:
        status_box.success(f"✅ Report complete (template mode — no API key). Also saved to `{saved_path}`")
    else:
        status_box.success(f"✅ Report complete. Also saved to `{saved_path}`")
    return {"mode": "full", "report": report_data, "saved_path": str(saved_path), "templates": using_templates}


# ── UI Components ─────────────────────────────────────────────────────────────

def render_sidebar() -> tuple[list[str], int, str, bool]:
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
        options=["full", "quick"],
        format_func=lambda x: "Full Report (LLM)" if x == "full" else "Quick Scan (no LLM)",
        help="Quick Scan skips summarization for instant results.",
    )

    no_cache = st.sidebar.checkbox("Bypass cache", help="Force fresh fetch instead of 6-hour cache")

    st.sidebar.divider()
    api_key_set = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if api_key_set:
        st.sidebar.success("✓ ANTHROPIC_API_KEY is set — high-quality LLM summaries available")
    else:
        st.sidebar.info(
            "ℹ️ ANTHROPIC_API_KEY not set — Full Report will use template-based "
            "summaries (snippet + impact framing). Set the key for AI-generated summaries."
        )

    return selected, days, mode, no_cache


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

    # Executive summary
    with st.container(border=True):
        st.markdown("### 📋 Executive Summary")
        for bullet in report.get("executive_summary", []):
            st.markdown(f"- {bullet}")

    # Recommended actions (top of page for visibility)
    with st.container(border=True):
        st.markdown("### 💡 Recommended Actions")
        for action in report.get("recommended_actions", []):
            st.markdown(f"- {action}")

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

    st.caption("⚡ Quick scan skips LLM summarization. Switch to 'Full Report' for sales-impact summaries and recommended actions.")


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

    selected_categories, days, mode, no_cache = render_sidebar()

    tab_run, tab_manage, tab_history, tab_help = st.tabs(["🚀 Run Report", "⚙️ Manage Categories", "📂 Report History", "❓ How to Use"])

    # ── Run tab ──────────────────────────────────────────────────────────────
    with tab_run:
        if not selected_categories:
            st.info("👈 Select one or more categories from the sidebar to begin.")
        else:
            run_btn = st.button("Generate Report", type="primary", use_container_width=False)
            if run_btn:
                for cat in selected_categories:
                    st.markdown(f"## {cat}")
                    status = st.empty()
                    result = run_pipeline(cat, days, mode, status, no_cache=no_cache)
                    status.empty()
                    if result is None:
                        continue
                    if result["mode"] == "full":
                        if result.get("templates"):
                            st.info(
                                "📋 **Template mode** — summaries built from article snippets + "
                                "impact-type framing. Set `ANTHROPIC_API_KEY` for AI-generated summaries."
                            )
                        render_full_report(result["report"])
                    else:
                        render_quick_report(result)
                    st.divider()

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
for their relevance to your Amazon product category, and generates a weekly
intelligence report with summaries and recommended actions.

---

### Quick Start (2 minutes)

1. **Select a category** from the sidebar dropdown (e.g. "Tires")
2. **Set the lookback period** — how many days of news to scan (7 is a good default)
3. **Choose a mode:**
   - **Full Report** — generates article summaries, executive summary, and recommended actions
   - **Quick Scan** — instant results showing scored articles without summaries
4. **Click "Generate Report"** and wait 10-30 seconds

---

### Understanding the Report

| Section | What it tells you |
|---|---|
| **Executive Summary** | The 3-5 most important themes this week at a glance |
| **Detailed Findings** | Each relevant article grouped by impact type, with a sales-impact summary |
| **Recommended Actions** | Specific steps to take based on this week's news |

**Impact Levels:**
- 🔴 **High (70-100):** Likely to directly affect pricing, availability, or demand
- 🟡 **Medium (40-69):** Worth monitoring — may affect the category within weeks
- 🟢 **Low (30-39):** Background signal — no immediate action needed

**Impact Types:**
- **Supply Chain** — shortages, factory closures, logistics disruptions
- **Pricing** — tariffs, cost increases, price wars
- **Demand** — trend shifts, viral moments, new tech adoption
- **Regulatory** — recalls, new laws, safety mandates
- **Competitive** — brand earnings, bankruptcies, product launches
- **Seasonal** — weather events, holiday demand shifts

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

All generated Full Reports are automatically saved. Go to **📂 Report History** to browse
and download past reports.

---

### FAQ

**Q: Why are some articles scored low or missing?**
The scorer uses keyword matching against your category's configured keywords. If an article
doesn't mention any of your terms, it won't score high. Edit the category keywords to improve coverage.

**Q: Can I run multiple categories at once?**
Yes — select multiple categories in the sidebar dropdown before clicking Generate Report.

**Q: Why does "Bypass cache" exist?**
Results are cached for 6 hours to avoid hitting news APIs repeatedly. Check "Bypass cache"
to force a fresh fetch — useful when you just added keywords or want the latest breaking news.

**Q: How do I get AI-generated summaries instead of templates?**
Set the `ANTHROPIC_API_KEY` environment variable before starting the app. Without it,
the tool uses template-based summaries built from article snippets — still useful, just less detailed.
        """)


if __name__ == "__main__":
    main()
