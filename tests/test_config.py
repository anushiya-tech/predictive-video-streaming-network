from pathlib import Path
import pytest
from src.utils.config import ConfigError, load_all

def test_load_all_reads_required_configs() -> None:
    config = load_all()
    assert config.section("labels")["horizon_seconds"] == 5
    assert len(config.sections) == 9

def test_load_all_rejects_missing_config(tmp_path: Path) -> None:
    (tmp_path / "data.yaml").write_text("seed: 42\nsplit: {train: 0.7, validation: 0.15, test: 0.15}\n")
    with pytest.raises(ConfigError, match="Required config file is missing"):
        load_all(tmp_path)

