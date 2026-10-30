"""Parse the PrizePicks projections feed into flat prop records."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from . import config
from .names import norm_team


@dataclass
class PPProp:
    pp_id: str
    player: str
    team: str | None
    position: str | None
    stat_type: str
    line: float
    odds_type: str
    start_time: str | None
    spec_key: str | None  # None if we do not price this stat
    allowed: str = "under_or_over"  # "over" means only More can be picked
    updated_at: str | None = None
    jersey: str | None = None
    opponent: str | None = None
    adjusted: bool = False  # PrizePicks pays a reduced multiplier on this line
    game_id: str | None = None

    @property
    def under_allowed(self) -> bool:
        return self.allowed != "over"


def _tonight(start: str | None) -> bool:
    """True if a game starts on today's date in US Eastern (the slate date)."""
    if not start:
        return False
    from datetime import datetime
    from zoneinfo import ZoneInfo
    et = ZoneInfo("America/New_York")
    days = (datetime.fromisoformat(start).astimezone(et).date() - datetime.now(et).date()).days
    return 0 <= days < config.SLATE_DAYS


def parse(pp: dict, log, keep_types=("standard",)) -> tuple[list[PPProp], dict]:
    """Return (props of the given odds types for the target game, diagnostics)."""
    included = pp.get("included") or []
    players = {}
    leagues = {}
    for inc in included:
        t = inc.get("type")
        a = inc.get("attributes") or {}
        if t in ("new_player", "player"):
            players[str(inc.get("id"))] = a
        elif t == "league":
            leagues[str(inc.get("id"))] = a.get("name")

    diag = Counter()
    league_names = Counter()
    out: list[PPProp] = []
    for row in pp.get("data") or []:
        a = row.get("attributes") or {}
        rel = row.get("relationships") or {}
        pid = str(((rel.get("new_player") or rel.get("player") or {}).get("data") or {}).get("id", ""))
        lid = str(((rel.get("league") or {}).get("data") or {}).get("id", ""))
        if lid in leagues:
            league_names[leagues[lid]] += 1
        player = players.get(pid, {})
        odds_type = str(a.get("odds_type") or "standard").lower()
        diag[f"odds_type:{odds_type}"] += 1
        if odds_type not in keep_types:
            continue
        if a.get("is_promo") or a.get("status") not in (None, "pre_game"):
            diag["skipped_promo_or_not_pregame"] += 1
            continue
        team = norm_team(player.get("team") or player.get("team_name"))
        if player.get("combo") or not team or "/" in team:
            diag["combo_player"] += 1  # two-player props: not priced
            continue
        if config.GAME_TEAMS is not None and team not in config.GAME_TEAMS:
            diag["other_game"] += 1
            continue
        if config.GAME_TEAMS is None and not _tonight(a.get("start_time")):
            diag["not_tonight"] += 1  # the feed also carries tomorrow's games
            continue
        stat = str(a.get("stat_type") or a.get("stat_display_name") or "")
        if "combo" in stat.lower():
            diag["combo_stat"] += 1
            continue
        if config.norm_stat(stat) in config.EXCLUDE_STATS:
            diag["excluded_stat"] += 1  # user choice: e.g. Fantasy Score is disregarded
            continue
        spec = config.PP_STAT_INDEX.get(config.norm_stat(stat))
        try:
            line = float(a.get("line_score"))
        except (TypeError, ValueError):
            diag["bad_line"] += 1
            continue
        out.append(PPProp(
            pp_id=str(row.get("id")),
            player=player.get("display_name") or player.get("name") or "?",
            team=team,
            position=player.get("position"),
            stat_type=stat,
            line=line,
            odds_type=odds_type,
            start_time=a.get("start_time"),
            spec_key=spec.key if spec else None,
            allowed=str(a.get("allowed_wager_types") or "under_or_over"),
            updated_at=a.get("updated_at"),
            jersey=player.get("jersey_number"),
            adjusted=bool(a.get("adjusted_odds")),
            opponent=(next((t for t in config.GAME_TEAMS if t != team), None) if config.GAME_TEAMS
                      else (str(a.get("description") or "").split()[0] or None)),
            game_id=a.get("game_id"),
        ))
    if league_names and not any(config.LEAGUE.upper() == n for n in league_names):
        log(f"PrizePicks: feed leagues look like {dict(league_names)}; league_id may be wrong.")
    diag["kept"] = len(out)
    return out, dict(diag)
