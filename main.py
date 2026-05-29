"""
Category News Intelligence Tool — CLI entry point.

Usage examples:
  python main.py --category "Tires" --days 7
  python main.py --category "Tires,Auto Parts" --days 7
  python main.py --all --days 7
  python main.py --add-category "Pet Food" --keywords "dog food,cat food,Purina,Blue Buffalo"
  python main.py --list-categories
  python main.py --category "Tires" --format json --output ./reports/
  python main.py --category "Tires" --format html
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import click
import yaml
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

# Ensure the project root is on sys.path when invoked as `python main.py`
sys.path.insert(0, str(Path(__file__).parent))

from src.fetcher.news_fetcher import fetch_news_for_category, setup_logging
from src.keywords.category_keywords import get_keywords
from src.models.article import Article, SummarizedArticle
from src.registry.categories import list_configured_categories
from src.report.report_generator import build_report_data, save_report
from src.scorer.relevance_scorer import score_articles
from src.summarizer.summarizer import summarize_articles

console = Console()

_CONFIG_PATH = "config.yaml"
_CATEGORY_CONFIG_DIR = "data/category_configs"


# ── Config helpers ────────────────────────────────────────────────────────────

def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _get_anthropic_client(config: dict) -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        console.print("[bold red]Error:[/] ANTHROPIC_API_KEY environment variable is not set.")
        raise SystemExit(1)
    return anthropic.Anthropic(api_key=api_key)


# ── Terminal display helpers ──────────────────────────────────────────────────

def _print_header(category: str, date_start: str, date_end: str, analyzed: int, relevant: int) -> None:
    console.print(
        Panel(
            f"[bold cyan]📰 Category News Report: {category}[/]\n"
            f"[dim]📅 {date_start} – {date_end}[/]\n"
            f"[dim]📊 {analyzed} articles scanned → [bold]{relevant}[/bold] relevant[/]",
            box=box.ROUNDED,
            expand=False,
        )
    )


def _print_articles(summarized: list[SummarizedArticle]) -> None:
    high = [a for a in summarized if a.impact_level == "High"]
    medium = [a for a in summarized if a.impact_level == "Medium"]
    low = [a for a in summarized if a.impact_level == "Low"]

    if high:
        console.print(f"\n[bold red]🔴 HIGH IMPACT ({len(high)} articles)[/]")
        for a in high:
            console.print(f"  • {a.title} [{a.source}]")

    if medium:
        console.print(f"\n[bold yellow]🟡 MEDIUM IMPACT ({len(medium)} articles)[/]")
        for a in medium:
            console.print(f"  • {a.title} [{a.source}]")

    if low:
        console.print(f"\n[bold green]🟢 LOW IMPACT ({len(low)} articles)[/]")
        for a in low:
            console.print(f"  • {a.title} [{a.source}]")


def _print_save_path(path: Path) -> None:
    console.print(f"\n[bold]📋 Report saved to:[/] [cyan]{path}[/]")


# ── Pipeline ──────────────────────────────────────────────────────────────────

def _run_pipeline(
    category: str,
    config: dict,
    client: anthropic.Anthropic,
    days: int,
    fmt: str,
    output_dir: str,
    no_cache: bool,
    quiet: bool,
) -> Path:
    """Run the full pipeline for one category. Returns the saved report path."""
    fetcher_cfg = config.get("fetcher", {})
    scorer_cfg = config.get("scorer", {})
    summarizer_cfg = config.get("summarizer", {})
    anthropic_cfg = config.get("anthropic", {})

    model = anthropic_cfg.get("model", "claude-haiku-4-5")
    cache_dir = fetcher_cfg.get("cache_dir", ".cache")
    threshold = scorer_cfg.get("threshold", 30)

    # 1 — Fetch
    if not quiet:
        console.print(f"  [dim]Fetching news for [cyan]{category}[/]…[/]")
    raw_articles_dicts = fetch_news_for_category(
        category,
        config=config,
        use_cache=not no_cache,
        days=days,
    )
    # Convert dicts to Article objects (fetcher returns dicts)
    raw_articles: list[Article] = []
    for d in raw_articles_dicts:
        raw_articles.append(Article(
            title=d.get("title", ""),
            source=d.get("source", ""),
            url=d.get("url", ""),
            published_date=d.get("published_date"),
            snippet=d.get("snippet", ""),
        ))

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

    # 4 — Terminal display
    if not quiet:
        from src.report.report_generator import _date_range_str
        date_start, date_end = _date_range_str(days)
        _print_header(category, date_start, date_end, len(raw_articles), len(summarized))
        _print_articles(summarized)

    # 5 — Build report data + LLM exec summary / actions
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

    # 6 — Save
    path = save_report(report_data, fmt=fmt, output_dir=output_dir)
    if not quiet:
        _print_save_path(path)

    return path


# ── CLI ───────────────────────────────────────────────────────────────────────

@click.group(invoke_without_command=True)
@click.option("--category", "-c", default=None, help="Category name(s), comma-separated")
@click.option("--all", "all_categories", is_flag=True, default=False, help="Process all registered categories")
@click.option("--days", default=7, show_default=True, help="Look back N days for news")
@click.option("--format", "fmt", default=None, type=click.Choice(["markdown", "html", "json"]), help="Output format")
@click.option("--output", default=None, help="Output directory (default: config report.output_dir)")
@click.option("--no-cache", is_flag=True, default=False, help="Bypass all local caches")
@click.option("--add-category", "add_cat", default=None, metavar="NAME", help="Register a new category")
@click.option("--keywords", default=None, help="Comma-separated keywords for --add-category")
@click.option("--list-categories", "list_cats", is_flag=True, default=False, help="List all registered categories")
@click.option("--quiet", "-q", is_flag=True, default=False, help="Suppress terminal output (still saves report)")
@click.pass_context
def cli(
    ctx: click.Context,
    category: str | None,
    all_categories: bool,
    days: int,
    fmt: str | None,
    output: str | None,
    no_cache: bool,
    add_cat: str | None,
    keywords: str | None,
    list_cats: bool,
    quiet: bool,
) -> None:
    """Category News Intelligence Tool — fetch, score, summarize, and report."""

    config = _load_config()
    setup_logging(config)

    effective_fmt = fmt or config.get("report", {}).get("default_format", "markdown")
    effective_output = output or config.get("report", {}).get("output_dir", "reports")

    # ── --list-categories ────────────────────────────────────────────────────
    if list_cats:
        cats = list_configured_categories(_CATEGORY_CONFIG_DIR)
        if not cats:
            console.print("[yellow]No categories configured yet.[/]")
        else:
            table = Table(title="Registered Categories", box=box.SIMPLE)
            table.add_column("Category", style="cyan")
            for cat in cats:
                table.add_row(cat)
            console.print(table)
        return

    # ── --add-category ───────────────────────────────────────────────────────
    if add_cat:
        if not keywords:
            console.print("[bold red]Error:[/] --keywords is required when using --add-category.")
            raise SystemExit(1)
        _add_category(add_cat, keywords)
        return

    # ── --all ────────────────────────────────────────────────────────────────
    if all_categories:
        cats = list_configured_categories(_CATEGORY_CONFIG_DIR)
        if not cats:
            console.print("[yellow]No categories found. Use --add-category to register one.[/]")
            return
        client = _get_anthropic_client(config)
        _run_batch(cats, config, client, days, effective_fmt, effective_output, no_cache, quiet)
        return

    # ── --category ───────────────────────────────────────────────────────────
    if category:
        cats = [c.strip() for c in category.split(",") if c.strip()]
        client = _get_anthropic_client(config)
        if len(cats) == 1:
            _run_pipeline(cats[0], config, client, days, effective_fmt, effective_output, no_cache, quiet)
        else:
            _run_batch(cats, config, client, days, effective_fmt, effective_output, no_cache, quiet)
        return

    # No action specified — show help
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


def _run_batch(
    categories: list[str],
    config: dict,
    client: anthropic.Anthropic,
    days: int,
    fmt: str,
    output_dir: str,
    no_cache: bool,
    quiet: bool,
) -> None:
    """Process multiple categories with a progress bar."""
    results: list[tuple[str, Path | None, str | None]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Processing categories…", total=len(categories))
        for cat in categories:
            progress.update(task, description=f"[cyan]{cat}[/]…")
            try:
                path = _run_pipeline(cat, config, client, days, fmt, output_dir, no_cache, quiet=True)
                results.append((cat, path, None))
            except Exception as exc:
                logger.error("Pipeline failed for %r: %s", cat, exc)
                results.append((cat, None, str(exc)))
            finally:
                progress.advance(task)

    # Summary table
    table = Table(title="Batch Results", box=box.SIMPLE)
    table.add_column("Category", style="cyan")
    table.add_column("Status")
    table.add_column("Report")
    for cat, path, err in results:
        if err:
            table.add_row(cat, "[red]✗ Failed[/]", err[:60])
        else:
            table.add_row(cat, "[green]✓ Done[/]", str(path))
    console.print(table)


def _add_category(name: str, keywords_csv: str) -> None:
    """Create a YAML config for a new category."""
    terms = [t.strip() for t in keywords_csv.split(",") if t.strip()]

    # Heuristic split: terms with spaces or all-caps are likely brands/supply-chain
    primary = [t for t in terms if " " not in t and not t.isupper()][:6]
    brands = [t for t in terms if t[0].isupper() and " " not in t][:8]
    supply_chain = [t for t in terms if any(w in t.lower() for w in ("recall", "shortage", "tariff", "supply"))]
    seasonal = [t for t in terms if any(w in t.lower() for w in ("season", "winter", "summer", "holiday"))]
    regulatory = [t for t in terms if any(w in t.lower() for w in ("regulation", "standard", "ban", "fda", "epa"))]

    # Whatever's left goes to primary
    assigned = set(primary + brands + supply_chain + seasonal + regulatory)
    remainder = [t for t in terms if t not in assigned]
    primary = list(dict.fromkeys(primary + remainder))[:6]

    data = {
        "category": name,
        "keywords": {
            "primary": primary or terms[:5],
            "brands": brands,
            "supply_chain": supply_chain,
            "seasonal": seasonal,
            "regulatory": regulatory,
        },
    }

    slug = name.lower().replace(" ", "_")
    path = Path(_CATEGORY_CONFIG_DIR) / f"{slug}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    console.print(f"[green]✓ Category [bold]{name!r}[/] added:[/] {path}")
    console.print(f"  Primary keywords: {', '.join(data['keywords']['primary'])}")
    if brands:
        console.print(f"  Brands: {', '.join(brands)}")
    console.print("\n[dim]Tip: Edit the YAML to refine keyword groups before running.[/]")


if __name__ == "__main__":
    cli()
