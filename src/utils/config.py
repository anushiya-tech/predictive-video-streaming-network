"""Validated loading of the project's YAML configuration files."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml

CONFIG_FILES = ("data.yaml", "features.yaml", "labels.yaml", "models.yaml", "adaptation.yaml", "qoe.yaml", "simulation.yaml", "video.yaml", "app.yaml")

class ConfigError(ValueError):
    """Raised when project configuration is missing or invalid."""

@dataclass(frozen=True)
class ProjectConfig:
    """All YAML configuration sections keyed by filename stem."""
    sections: dict[str, dict[str, Any]]

    def section(self, name: str) -> dict[str, Any]:
        """Return a named config section or raise a clear error."""
        try:
            return self.sections[name]
        except KeyError as error:
            raise ConfigError(f"Unknown config section: {name}") from error

def _load_file(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigError(f"Cannot read config file {path}: {error}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"Invalid YAML in {path}: {error}") from error
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a YAML mapping")
    if raw.get("seed") != 42:
        raise ConfigError(f"{path}: 'seed' must be the integer 42")
    return raw

def load_all(config_dir: str | Path | None = None) -> ProjectConfig:
    """Load every required config file and validate baseline invariants."""
    directory = Path(config_dir) if config_dir is not None else Path(__file__).resolve().parents[2] / "configs"
    if not directory.is_dir():
        raise ConfigError(f"Config directory does not exist: {directory}")
    sections: dict[str, dict[str, Any]] = {}
    for filename in CONFIG_FILES:
        path = directory / filename
        if not path.is_file():
            raise ConfigError(f"Required config file is missing: {path}")
        sections[path.stem] = _load_file(path)
    split = sections["data"].get("split")
    if not isinstance(split, dict) or abs(sum(split.values()) - 1.0) > 1e-9:
        raise ConfigError("configs/data.yaml: split proportions must sum to 1.0")
    if sections["labels"].get("horizon_seconds", 0) <= 0:
        raise ConfigError("configs/labels.yaml: horizon_seconds must be positive")
    return ProjectConfig(sections=sections)

