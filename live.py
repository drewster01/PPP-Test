"""Live board: serves web/index.html for the NBA and NFL slates and reruns both models on a schedule.

    python live.py            # then open http://127.0.0.1:8766

Each league runs in its own process (config is read per league at import), every
LIVE_REFRESH_MIN minutes until the end of the day's slate (US Eastern), and on demand
from /run?league=nba|nfl.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
INDEX = ROOT / "web" / "index.html"
DATA = ROOT / "data"
LEAGUES = ("nba", "nfl")
PAGES = ("picks", "leans", "results")
PORT = 8766
REFRESH_MIN = 10
BOARDS = {"board.json", "board_leans.json", "board_results.json", "board_plays.json", "board_mores.json"}
_locks = {lg: threading.Lock() for lg in LEAGUES}
_state = {lg: {"running": False, "last_run": None, "last_log": ""} for lg in LEAGUES}


def run_model(league: str) -> None:
    if not _locks[league].acquire(blocking=False):
        return  # a run is already in progress
    try:
        _state[league]["running"] = True
        proc = subprocess.run([sys.executable, str(ROOT / "props_edge.py"), "--league", league, "--trials", "100000"],
                              cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        _state[league]["last_log"] = (proc.stdout + proc.stderr)[-4000:]
        _state[league]["last_run"] = datetime.now(timezone.utc).isoformat()
        print(f"[{datetime.now():%H:%M:%S}] {league} model run finished (exit {proc.returncode})")
    finally:
        _state[league]["running"] = False
        _locks[league].release()


def scheduler() -> None:
    et = ZoneInfo("America/New_York")
    while True:
        now = datetime.now(et)
        if now.hour >= 9:  # PrizePicks posts the day's board in the morning; stop at midnight ET
            for lg in LEAGUES:
                run_model(lg)
        time.sleep(REFRESH_MIN * 60)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path, _, query = self.path.partition("?")
        parts = [p for p in path.split("/") if p]
        # old single-league links (/leans, /mores, ...) go to the NBA board
        old = {"": "picks", "mores": "picks", "plays": "picks", "leans": "leans", "results": "results"}
        if len(parts) <= 1 and (parts[0] if parts else "") in old:
            self._send(302, b"", "text/plain", {"Location": f"/nba/{old[parts[0] if parts else '']}"})
        elif len(parts) == 1 and parts[0] in LEAGUES:
            self._send(302, b"", "text/plain", {"Location": f"/{parts[0]}/picks"})
        elif len(parts) == 2 and parts[0] in LEAGUES and parts[1] in PAGES:
            self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
        elif len(parts) == 2 and parts[0] in LEAGUES and parts[1] in BOARDS:
            f = DATA / parts[0] / parts[1]
            self._send(200, f.read_bytes() if f.exists() else b"{}", "application/json")
        elif parts == ["status"]:
            self._send(200, json.dumps(_state).encode(), "application/json")
        elif parts == ["run"]:
            lg = query.partition("league=")[2] or "nba"
            if lg in LEAGUES:
                threading.Thread(target=run_model, args=(lg,), daemon=True).start()
            self._send(202, b'{"started": true}', "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def log_message(self, *args) -> None:  # quiet console
        pass


def main() -> None:
    threading.Thread(target=scheduler, daemon=True).start()
    # --lan: also reachable from other devices on the home network (phone on the same Wi-Fi)
    host = "0.0.0.0" if "--lan" in sys.argv else "127.0.0.1"
    server = ThreadingHTTPServer((host, PORT), Handler)
    print(f"Board live at http://127.0.0.1:{PORT}" + (" and on your network (see your PC's Wi-Fi IP)" if host != "127.0.0.1" else "")
          + "  (Ctrl+C to stop)")
    server.serve_forever()


if __name__ == "__main__":
    main()
