import subprocess
import time
import urllib.request
import json
import sys

def main():
    print("=== 1. Starting server via python run.py ===")
    proc = subprocess.Popen([sys.executable, "run.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        connected = False
        for i in range(15):
            time.sleep(1)
            try:
                with urllib.request.urlopen("http://127.0.0.1:8000/api/simulation/scenarios", timeout=2) as r:
                    if r.status == 200:
                        scenarios = json.loads(r.read().decode("utf-8"))
                        print("Server online! Scenarios:", scenarios)
                        connected = True
                        break
            except Exception:
                continue
        assert connected, "Server failed to start within 15s"

        # Check root HTML
        with urllib.request.urlopen("http://127.0.0.1:8000/", timeout=2) as r:
            assert r.status == 200
            html = r.read().decode("utf-8")
            assert "Predictive Video Streaming Network" in html
            print("Root endpoint: 200 OK, HTML length:", len(html))

        # Check video status
        with urllib.request.urlopen("http://127.0.0.1:8000/api/video/status", timeout=2) as r:
            assert r.status == 200
            vid = json.loads(r.read().decode("utf-8"))
            assert vid["ffmpeg_available"] is False
            print("Video status: 200 OK, ffmpeg_available: False")

        # Check baselines
        with urllib.request.urlopen("http://127.0.0.1:8000/api/baselines", timeout=5) as r:
            assert r.status == 200
            base = json.loads(r.read().decode("utf-8"))
            assert len(base) == 10
            print("Baselines endpoint: 200 OK, entries:", len(base))

        # Test all 6 scenarios over HTTP
        for sc in scenarios:
            req = urllib.request.Request(
                "http://127.0.0.1:8000/api/simulation/scenario",
                data=json.dumps({"scenario": sc}).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=30) as r:
                assert r.status == 200

                res = json.loads(r.read().decode("utf-8"))
                assert res["scenario"] == sc
                assert res["ticks"] == 60

            with urllib.request.urlopen("http://127.0.0.1:8000/api/network/history", timeout=2) as r:
                assert r.status == 200
                hist = json.loads(r.read().decode("utf-8"))
                assert len(hist) == 60

            with urllib.request.urlopen("http://127.0.0.1:8000/api/network/current", timeout=2) as r:
                assert r.status == 200
                curr = json.loads(r.read().decode("utf-8"))
                assert curr["tick"] == 59
                assert curr["quality"] in {"360p", "480p", "720p", "1080p"}

            with urllib.request.urlopen("http://127.0.0.1:8000/api/prediction/current", timeout=2) as r:
                assert r.status == 200
                pred = json.loads(r.read().decode("utf-8"))
                assert pred["network_condition"] in {"Stable", "Improving", "Degrading"}
                assert pred["buffer_risk"] in {"Low", "Medium", "High"}
                assert 0.0 <= pred["network_confidence"] <= 1.0
                assert 0.0 <= pred["buffer_confidence"] <= 1.0

            q = curr["quality"]
            risk = pred["buffer_risk"]
            cond = pred["network_condition"]
            print(f"Scenario HTTP test: {sc} -> 60 ticks, quality={q}, risk={risk}, condition={cond}")

        print("=== ALL RUNTIME HTTP API AND SCENARIO CHECKS PASSED ===")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("Server cleanly terminated.")

if __name__ == "__main__":
    main()
