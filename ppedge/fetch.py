"""Network access: SportsGameOdds (keyed) and the public PrizePicks board.

Rules enforced here:
  * the API key is read from SGO_KEY and never printed, logged or written;
  * blocked requests (401/403/429) are reported, not retried or evaded;
  * request count is capped and spaced for the free tier;
  * recent raw files are reused so iterating on the code costs no requests.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from . import config

# each league keeps its own data, logs and board
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / config.LEAGUE
RAW_SGO = DATA_DIR / "raw_sgo.json"
RAW_PP = DATA_DIR / "raw_pp.json"
RAW_META = DATA_DIR / "raw_meta.json"
SNAP_DIR = DATA_DIR / "snapshots"


class FetchError(RuntimeError):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _load_meta() -> dict:
    try:
        return json.loads(RAW_META.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_meta(meta: dict) -> None:
    RAW_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _fresh(path: Path, meta_key: str, meta: dict) -> bool:
    if not path.exists() or meta_key not in meta:
        return False
    age = _now() - datetime.fromisoformat(meta[meta_key])
    return age.total_seconds() < config.CACHE_MAX_AGE_MIN * 60


def _describe_block(resp: requests.Response, source: str) -> str:
    hint = {
        401: "key missing or invalid",
        403: "request refused (blocked or not allowed on this tier)",
        429: "rate limit or monthly quota reached",
    }.get(resp.status_code, "unexpected response")
    return f"{source}: HTTP {resp.status_code} ({hint})"


def fetch_sgo(log) -> dict | None:
    key = os.environ.get("SGO_KEY", "").strip()
    if not key:
        log("SportsGameOdds: SGO_KEY is not set. Run set_key.ps1 (or set the variable) and retry.")
        return None
    headers = {"x-api-key": key, "User-Agent": config.BROWSER_UA, "Accept": "application/json"}
    params = dict(config.SGO_PARAMS)
    # Only tonight's game: each event returned counts against the monthly quota.
    kick = datetime.fromisoformat(config.KICKOFF_ISO).astimezone(timezone.utc)
    params["startsAfter"] = (kick - timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
    params["startsBefore"] = (kick + timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
    retried_without_window = False
    events: list = []
    pages = 0
    last = 0.0
    while pages < config.SGO_MAX_PAGES:
        wait = config.SGO_MIN_SECONDS_BETWEEN_REQUESTS - (time.monotonic() - last)
        if pages and wait > 0:
            time.sleep(wait)
        last = time.monotonic()
        try:
            resp = requests.get(config.SGO_EVENTS_URL, params=params, headers=headers, timeout=30)
        except requests.RequestException as exc:
            log(f"SportsGameOdds: network error ({type(exc).__name__}); continuing without it.")
            break
        pages += 1
        if resp.status_code == 400 and not retried_without_window:
            # date-window format not accepted: fall back to the brief's plain query
            params.pop("startsAfter", None)
            params.pop("startsBefore", None)
            retried_without_window = True
            log("SportsGameOdds: date filter rejected; retrying with the plain query.")
            continue
        if resp.status_code != 200:
            log(_describe_block(resp, "SportsGameOdds") + "; continuing with what is available.")
            break
        body = resp.json()
        batch = body.get("data") or []
        events.extend(batch)
        # stop as soon as the target game is in hand
        if any(_is_target_game(e) for e in batch):
            break
        cursor = body.get("nextCursor")
        if not cursor:
            break
        params["cursor"] = cursor
    log(f"SportsGameOdds: {pages} request(s), {len(events)} event(s).")
    if not events:
        return None
    return {"success": True, "data": events}


def _is_target_game(event: dict) -> bool:
    teams = event.get("teams") or {}
    found = set()
    for side in ("home", "away"):
        names = (teams.get(side) or {}).get("names") or {}
        vals = {str(v).upper() for v in names.values() if v}
        vals.add(str((teams.get(side) or {}).get("teamID", "")).upper().replace("_", " "))
        for abbr, aliases in config.TEAM_ALIASES.items():
            if vals & aliases or any(a in v for v in vals for a in aliases if len(a) > 3):
                found.add(abbr)
    return set(config.GAME_TEAMS) <= found


def fetch_pp(log) -> dict | None:
    headers = {"User-Agent": config.BROWSER_UA, "Accept": "application/json"}
    try:
        resp = requests.get(config.PP_URL, params=config.PP_PARAMS, headers=headers, timeout=30)
    except requests.RequestException as exc:
        log(f"PrizePicks: network error ({type(exc).__name__}).")
        return None
    if resp.status_code != 200:
        log(_describe_block(resp, "PrizePicks")
            + ". You can save the board JSON from your browser to data/raw_pp.json and rerun with --offline.")
        return None
    body = resp.json()
    log(f"PrizePicks: {len(body.get('data') or [])} projection(s).")
    return body


def get_data(log, refresh: bool = False, offline: bool = False) -> tuple[dict | None, dict | None, dict]:
    """Return (sgo, pp, meta). Uses cached raw files when fresh or offline."""
    DATA_DIR.mkdir(exist_ok=True)
    meta = _load_meta()

    def load(path: Path):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    sgo = pp = None
    if offline or (not refresh and _fresh(RAW_SGO, "sgo_fetched_at", meta)):
        sgo = load(RAW_SGO)
        if sgo is not None:
            log(f"SportsGameOdds: using saved data from {meta.get('sgo_fetched_at', 'unknown time')}.")
    if sgo is None and not offline:
        sgo = fetch_sgo(log)
        if sgo is not None:
            RAW_SGO.write_text(json.dumps(sgo), encoding="utf-8")
            meta["sgo_fetched_at"] = _now().isoformat()

    if offline or (not refresh and _fresh(RAW_PP, "pp_fetched_at", meta)):
        pp = load(RAW_PP)
        if pp is not None:
            log(f"PrizePicks: using saved data from {meta.get('pp_fetched_at', 'unknown time')}.")
    if pp is None and not offline:
        pp = fetch_pp(log)
        if pp is not None:
            RAW_PP.write_text(json.dumps(pp), encoding="utf-8")
            meta["pp_fetched_at"] = _now().isoformat()

    _save_meta(meta)
    return sgo, pp, meta


def cached(name: str, fetcher, log, refresh: bool = False, offline: bool = False):
    """Generic cache wrapper for extra sources (data/raw_<name>.json)."""
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"raw_{name}.json"
    meta = _load_meta()
    key = f"{name}_fetched_at"
    if offline or (not refresh and _fresh(path, key, meta)):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            if offline:
                return None
    data = fetcher(log)
    if data is not None:
        path.write_text(json.dumps(data), encoding="utf-8")
        meta = _load_meta()
        meta[key] = _now().isoformat()
        _save_meta(meta)
    return data


def snapshot(files: list[Path]) -> Path:
    """Copy this run's raw inputs and outputs into a timestamped folder."""
    stamp = _now().strftime("%Y%m%dT%H%M%SZ")
    dest = SNAP_DIR / stamp
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        if f.exists():
            shutil.copy2(f, dest / f.name)
    return dest
