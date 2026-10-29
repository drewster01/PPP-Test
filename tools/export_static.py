"""Export the board as one self-contained page (both leagues, every tab) to view on a phone.

    python tools/export_static.py [out.html]

The live boards are embedded in the page, so it works with no server. It is a snapshot:
rerun to refresh it.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
LEAGUES = ("nba", "nfl")
FILES = ("board.json", "board_leans.json", "board_results.json")
# card fields the page never reads on the Picks tab; dropping them keeps the snapshot small
DROP = ("devig_range", "quotes", "dist", "books", "comparison")


def load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "phone_board.html"
    boards = {}
    for lg in LEAGUES:
        for f in FILES:
            doc = load(ROOT / "data" / lg / f)
            if doc is None:
                continue
            for p in doc.get("props", []):
                for k in DROP:
                    p.pop(k, None)
            boards[f"/{lg}/{f}"] = doc
    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    style = re.search(r"<style>.*?</style>", html, re.S).group(0)
    body = re.search(r"<body>(.*)</body>", html, re.S).group(1)
    stamp = datetime.now(ZoneInfo("America/New_York")).strftime("%a %b %d, %I:%M %p ET").replace(" 0", " ")
    # the page script reads window.STATIC_BOARDS before anything else, so the data goes first
    data = json.dumps(boards, separators=(",", ":"), default=str).replace("</", "<\\/")
    body = body.replace("<script>", f"<script>window.STATIC_BOARDS = {data};</script>\n<script>", 1)
    body = body.replace("<footer>", f"<footer>Snapshot from {stamp}. ", 1)
    style = style.replace("<style>", "<style>\n:root{color-scheme:dark}", 1)
    out.write_text(f"<title>PrizePicks Edge Board</title>\n{style}\n{body}", encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
