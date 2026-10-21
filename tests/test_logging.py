import json
from src.utils.logging import get_logger

def test_logger_writes_json_lines(tmp_path) -> None:
    logger = get_logger("phase_zero_test", tmp_path)
    logger.info("configuration loaded")
    for handler in logger.handlers:
        handler.flush()
    record = json.loads((tmp_path / "phase_zero_test.jsonl").read_text().strip())
    assert record["message"] == "configuration loaded"
    assert record["level"] == "INFO"

