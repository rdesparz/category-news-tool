"""Category registry — discovers available YAML configs."""

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def list_configured_categories(config_dir: str = "data/category_configs") -> list[str]:
    """Return display names of all categories with YAML configs."""
    path = Path(config_dir)
    if not path.exists():
        return []
    categories = []
    for f in sorted(path.glob("*.yaml")):
        try:
            with open(f) as fh:
                data = yaml.safe_load(fh)
            name = data.get("category") or f.stem.replace("_", " ").title()
            categories.append(name)
        except Exception:
            logger.warning("Could not parse %s", f)
    return categories
