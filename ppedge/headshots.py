"""Player headshots from ESPN's public team rosters (no key). Cached for a day."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import requests

from . import config
from .fetch import DATA_DIR
from .names import norm_name, short_key

CACHE = DATA_DIR / "headshots.json"
SPORT_PATH = {"nfl": "football/nfl", "nba": "basketball/nba"}[config.LEAGUE]
URL = "https://site.api.espn.com/apis/site/v2/sports/" + SPORT_PATH + "/teams/{}/roster"
# PrizePicks abbreviation -> ESPN's, where they differ
ESPN_ABBR = {"GSW": "gs", "NOP": "no", "NYK": "ny", "SAS": "sa", "UTA": "utah", "WAS": "wsh"}


def load(log, teams=None) -> dict:
    """Return {"TEAM|normalized name" or "TEAM|short key": headshot url}."""
    teams = sorted(set(teams or config.GAME_TEAMS or ()))
    try:
        doc = json.loads(CACHE.read_text(encoding="utf-8"))
        fresh = datetime.now(timezone.utc) - datetime.fromisoformat(doc["fetched_at"]) < timedelta(days=1)
        if fresh and set(teams) <= set(doc.get("teams", [])):
            return doc["photos"]
    except (OSError, ValueError, KeyError):
        pass
    photos = {}
    for team in teams:
        try:
            r = requests.get(URL.format(ESPN_ABBR.get(team, team.lower())), timeout=20,
                             headers={"User-Agent": config.BROWSER_UA})
            if r.status_code != 200:
                log(f"ESPN roster {team}: HTTP {r.status_code}; using initials.")
                continue
            body = r.json()
            groups = body.get("athletes", [])
            # NFL groups players by unit ({"items": [...]}); NBA returns a flat list
            athletes = [a for g in groups for a in g.get("items", [])] if groups and "items" in groups[0] else groups
            for a in athletes:
                href = (a.get("headshot") or {}).get("href")
                name = a.get("displayName") or a.get("fullName")
                if href and name:
                    photos[f"{team}|{norm_name(name)}"] = href
                    photos.setdefault(f"{team}|{short_key(name)}", href)
        except requests.RequestException as exc:
            log(f"ESPN roster {team}: {type(exc).__name__}; using initials.")
    if photos:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps({"fetched_at": datetime.now(timezone.utc).isoformat(),
                                     "teams": teams, "photos": photos}), encoding="utf-8")
    return photos


def find(photos: dict, player: str, team: str | None) -> str | None:
    return photos.get(f"{team}|{norm_name(player)}") or photos.get(f"{team}|{short_key(player)}")
