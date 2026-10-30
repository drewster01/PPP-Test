"""Football defensive factors (NFL and college) for strength of schedule.

What each defense allows per game in rushing and passing yards, divided by the league average, from
TeamRankings' public opponent-stat tables (this season's column), keyed by ESPN team abbreviation so
they line up with ESPN game logs. Cached for a day in data/<league>/defense.json.
"""
from __future__ import annotations

import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TR = {"nfl": "nfl", "cfb": "college-football"}
ESPN = {"nfl": "football/nfl", "cfb": "football/college-football"}
PAGES = {"rush": "opponent-rushing-yards-per-game", "pass": "opponent-passing-yards-per-game"}
# PrizePicks stat -> which defensive factor moves it
FAMILY = {
    "Rush Yards": "rush", "Rush Attempts": None, "Longest Rush": "rush", "Rush TDs": "rush",
    "Receiving Yards": "pass", "Receptions": "pass", "Rec Targets": None, "Longest Reception": "pass",
    "Pass Yards": "pass", "Pass Completions": "pass", "Pass TDs": "pass", "Longest Completion": "pass",
    "Completion Percentage": "pass", "Rush+Rec Yds": "rush", "Pass+Rush Yds": "pass",
}


def _get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower().replace("state", "st").replace("&", "and"))


def factors(league: str) -> dict[str, dict[str, float]]:
    """{ESPN abbreviation: {"rush": f, "pass": f}}; empty if the sources are unreachable."""
    path = ROOT / "data" / league / "defense.json"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if (datetime.now(timezone.utc) - datetime.fromisoformat(doc["fetched_at"])).total_seconds() < 86400:
            return doc["factors"]
    except (OSError, ValueError, KeyError):
        pass
    try:
        teams = json.loads(_get(f"https://site.api.espn.com/apis/site/v2/sports/{ESPN[league]}/teams?limit=1000"
                                + ("&groups=80" if league == "cfb" else "")))
        teams = [t["team"] for t in teams["sports"][0]["leagues"][0]["teams"]]
        by_name = {}
        for t in teams:
            for k in ("location", "shortDisplayName", "displayName", "name"):
                if t.get(k):
                    by_name.setdefault(_norm(t[k]), t["abbreviation"])
        out: dict[str, dict[str, float]] = {}
        for fam, page in PAGES.items():
            html = _get(f"https://www.teamrankings.com/{TR[league]}/stat/{page}")
            rows = re.findall(r'data-sort="([^"]+)"><a[^>]*>[^<]*</a></td>\s*<td class="text-right" data-sort="([\d.]+)"', html)
            vals = {name: float(v) for name, v in rows}
            avg = sum(vals.values()) / len(vals)
            for name, v in vals.items():
                ab = by_name.get(_norm(name)) or by_name.get(_norm(name.replace("St", "State")))
                if ab:
                    out.setdefault(ab, {})[fam] = v / avg
    except Exception:
        return {}
    path.write_text(json.dumps({"fetched_at": datetime.now(timezone.utc).isoformat(), "factors": out}), encoding="utf-8")
    return out


if __name__ == "__main__":
    import sys
    f = factors(sys.argv[1])
    print(len(f), "teams;", sorted(f.items(), key=lambda kv: kv[1].get("pass", 1))[:3])
