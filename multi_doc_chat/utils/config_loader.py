from pathlib import Path

import yaml


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "config.yaml"
)


def load_config(config_path=None):
    """Load the application YAML configuration."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}
