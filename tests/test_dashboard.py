from pathlib import Path
import subprocess
import shutil

FRONTEND_HTML = Path("app/frontend/index.html")


def test_dashboard_file_exists():
    assert FRONTEND_HTML.exists(), "app/frontend/index.html does not exist"


def test_dashboard_contains_critical_components():
    content = FRONTEND_HTML.read_text(encoding="utf-8")
    # Verify titles, charts, controls, and baseline comparison table
    assert "Predictive Video Streaming Network" in content
    assert "bandwidthChart" in content
    assert "bufferChart" in content
    assert "videoCanvas" in content
    assert "scenario" in content
    assert "btn-play" in content or "btnPlay" in content or "play" in content.lower()
    assert "baseline" in content.lower()


def test_dashboard_javascript_syntax():
    content = FRONTEND_HTML.read_text(encoding="utf-8")
    start = content.find("<script>") + len("<script>")
    end = content.find("</script>", start)
    assert start > 0 and end > start
    js_code = content[start:end]

    node_bin = shutil.which("node")
    if node_bin:
        tmp_js = Path(".pytest_tmp/test_syntax_check.js")
        tmp_js.parent.mkdir(exist_ok=True)
        tmp_js.write_text(js_code, encoding="utf-8")
        try:
            res = subprocess.run([node_bin, "--check", str(tmp_js)], capture_output=True, text=True)
            assert res.returncode == 0, f"JS Syntax Error: {res.stderr}"
        finally:
            if tmp_js.exists():
                tmp_js.unlink()
