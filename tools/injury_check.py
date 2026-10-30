"""Check every NBA lean against the latest official NBA injury report.

    python tools/injury_check.py

Last night every losing slip leg was turned by news about the player or a teammate after entry. This
reads the newest report (published every half hour), then for each of today's leans flags:
- the player himself (Out: the lean is cut; Questionable/Doubtful: flagged),
- the teammate the research named in "depends_on",
- any rotation teammate (20+ minutes a game) who is questionable or newly out.
Flags are saved on the lean and shown on the board; pending news costs a little confidence.
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))
import leans  # noqa: E402
import nbastats  # noqa: E402
from pypdf import PdfReader  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "data" / "nba" / "injury_report.pdf"
TEAMS = {"Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN", "Charlotte Hornets": "CHA",
         "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE", "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN",
         "Detroit Pistons": "DET", "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
         "LA Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM", "Miami Heat": "MIA",
         "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP",
         "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL",
         "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR",
         "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR", "Utah Jazz": "UTA",
         "Washington Wizards": "WAS"}
ALIAS = {"WSH": "WAS", "GS": "GSW", "NY": "NYK", "NO": "NOP", "SA": "SAS", "UTAH": "UTA"}
PENDING_PENALTY = 3  # points off a lean for each unresolved status it depends on
ET = ZoneInfo("America/New_York")


def fetch_latest() -> str | None:
    """Download the newest report published today (tries half-hour slots, newest first)."""
    now = datetime.now(ET).replace(second=0, microsecond=0)
    slot = now.replace(minute=30 if now.minute >= 30 else 0)
    for i in range(48):
        t = slot - timedelta(minutes=30 * i)
        url = (f"https://ak-static.cms.nba.com/referee/injury/Injury-Report_{t:%Y-%m-%d}_"
               f"{(t.hour % 12) or 12:02d}_{t:%M}{'PM' if t.hour >= 12 else 'AM'}.pdf")
        try:
            data = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}),
                                          timeout=15).read()
        except Exception:
            continue
        REPORT.write_bytes(data)
        return url
    return None


def parse() -> dict[str, list[tuple[str, str, str]]]:
    """{team: [(player, status, reason kind)]} from the saved report."""
    text = " ".join(" ".join(p.extract_text() or "" for p in PdfReader(str(REPORT)).pages).split())
    spans = sorted((m.start(), ab) for full, ab in TEAMS.items() for m in re.finditer(re.escape(full), text))
    out: dict[str, list] = {}
    for i, (start, ab) in enumerate(spans):
        chunk = text[start:spans[i + 1][0] if i + 1 < len(spans) else len(text)]
        for m in re.finditer(r"([A-Z][A-Za-z.'\-]+(?: (?:III|II|IV|Jr\.|Sr\.))?), ([A-Z][A-Za-z.'\-]+) "
                             r"(Out|Questionable|Doubtful|Probable|Available) (G League)?", chunk):
            out.setdefault(ab, []).append((f"{m.group(2)} {m.group(1)}", m.group(3), m.group(4) or ""))
    return out


def main() -> None:
    url = fetch_latest()
    if not url:
        sys.exit("No injury report found for today.")
    report = parse()
    logs_now, logs_last = nbastats.game_logs(2027), nbastats.game_logs(2026)

    def mpg(name):
        g = logs_now.get(nbastats.norm(name)) or logs_last.get(nbastats.norm(name)) or []
        return sum(x["minutes"] for x in g) / len(g) if g else 0

    status = {nbastats.norm(n): (s, t) for t, rows in report.items() for n, s, kind in rows if kind != "G League"}
    path = ROOT / "data" / "nba" / "board_leans.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    for lean in doc["leans"]:
        team = ALIAS.get(lean.get("team"), lean.get("team"))
        flags, pending = [], 0
        mine = status.get(nbastats.norm(lean["player"]))
        if mine:
            if mine[0] == "Out":
                lean["cut"] = True
                flags.append("ruled OUT")
            elif mine[0] in ("Questionable", "Doubtful"):
                flags.append(f"he is {mine[0].lower()}")
                pending += 1
        dep = (lean.get("depends_on") or "").lower()
        for name, (s, t) in status.items():
            if t != team or name == nbastats.norm(lean["player"]) or s in ("Available", "Probable"):
                continue
            named = name.split()[-1] in dep
            if named or (s in ("Questionable", "Doubtful") and mpg(name) >= 20):
                flags.append(f"{name.title()} {s.lower()}" + (" (named dependency)" if named else ""))
                pending += s in ("Questionable", "Doubtful")
        lean["injury_flags"] = flags
        lean["injury_penalty"] = PENDING_PENALTY * pending
        lean["injury_checked"] = url.rsplit("_", 2)[-2] + " " + url.rsplit("_", 1)[-1][:-4]
    path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    leans.rescore("nba")
    print("report:", url)
    for lean in json.loads(path.read_text(encoding="utf-8"))["leans"]:
        if lean.get("injury_flags"):
            print(f"  {lean['player']} {lean['side']} {lean.get('current_line')} {lean['stat']}: "
                  f"{'; '.join(lean['injury_flags'])}{'  -> CUT' if lean.get('cut') else ''}")


if __name__ == "__main__":
    main()
