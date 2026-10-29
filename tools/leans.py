"""Research leans: daily board, grading against box scores, and a model that learns from results.

    python tools/leans.py new   <league>                       # archive the old board, start today's
    python tools/leans.py add   <league> <batch.json> <label> [slate note ...]
    python tools/leans.py grade <league> <YYYYMMDD>            # grade that day's archived leans
    python tools/leans.py void  <league> <YYYYMMDD> "<player>" [note]  # player left early: don't count it
    python tools/leans.py model <league>                       # print the record, re-score today's leans
    python tools/leans.py vet   <league> <vet.json>            # apply a minutes-risk / late-news pass
    python tools/leans.py brief <league>                       # the record as instructions for researchers

Every graded lean goes into data/<league>/leans_history.json. The model turns that history into
hit rates for each feature of a lean (stat type and side, how much the projection clears the line,
the researcher's confidence, payout, minutes risk) and blends them with the research estimate, so
what keeps losing gets marked down and what keeps winning gets marked up. Leans where the player
left early (injury, ejection, blowout) are shown but never counted.
"""
from __future__ import annotations

import json
import math
import os
import sys
import unicodedata
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hitrate  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ET = ZoneInfo("America/New_York")
SPORT = {"nba": "basketball/nba", "nfl": "football/nfl"}

# How much one lean type's record moves its confidence. Records are shrunk toward the side's
# record (prior weight GROUP_PRIOR), and the side's record toward 50% (SIDE_PRIOR), so a handful
# of results nudges the number and a long record moves it a lot.
SIDE_PRIOR = 10
GROUP_PRIOR = 6
MODEL_WEIGHT = 0.4  # share of the final confidence that comes from the track record
CONF_RANGE = (45, 72)
SHORT_SHARE = 0.6  # under this share of his usual minutes, the player left early and the lean doesn't count


def norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = s.replace(".", "").replace("'", "").replace("-", " ")
    return " ".join(w for w in s.split() if w not in ("jr", "sr", "ii", "iii", "iv"))


def ddir(league: str) -> Path:
    return ROOT / "data" / league


def load(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def save(path: Path, doc) -> None:
    path.write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")


def today() -> str:
    return datetime.now(ET).strftime("%Y%m%d")


# ---------------------------------------------------------------- lean types
def group(league: str, stat: str, line: float) -> str:
    """Coarse lean type the model keeps a record for."""
    s = stat.lower()
    if league == "nba":
        if s in ("turnovers", "steals", "blocked shots", "personal fouls", "blks+stls"):
            return "low-count"
        if "3-pt attempted" in s or "3-pt made" in s:
            return "threes"
        if "free throw" in s:
            return "free throws"
        if "rebound" in s:
            return "rebounds"
        if "fg " in s or "two pointers" in s:
            return "shot volume"
        if s == "assists":
            return "assists"
        return "points/combos"
    if "longest" in s:
        return "longest play"
    if "td" in s or s == "int" or "sacks" in s:
        return "touchdowns/turnovers"
    if "kicking" in s or "fg made" in s or "punt" in s:
        return "kicking"
    if "attempts" in s or "completions" in s or "targets" in s or "receptions" in s:
        return "volume"
    if "tackles" in s:
        return "defense"
    return "yards"


def cushion(lean: dict) -> str:
    """How far the research projection clears the line, toward the lean's side, as a share of the line."""
    proj, line = lean.get("projection"), lean.get("line")
    if proj is None or line is None:
        return "no projection"
    room = (proj - line) if lean["side"] == "More" else (line - proj)
    share = room / max(abs(line), 1)
    return "thin cushion" if share < 0.10 else "solid cushion" if share < 0.25 else "big cushion"


def conf_band(lean: dict) -> str:
    c = lean["confidence"]
    return "research 55-57" if c <= 57 else "research 58-60" if c <= 60 else "research 61+"


# Every feature the model keeps a record for: name -> function(league, lean) -> bucket.
# Each bucket's record moves the confidence away from the side's record; WEIGHTS sets how far.
FEATURES = {
    "type": lambda lg, l: f"{group(lg, l['stat'], l['line'])} {l['side']}",
    "cushion": lambda lg, l: f"{cushion(l)} {l['side']}",
    "research": lambda lg, l: conf_band(l),
    "payout": lambda lg, l: "reduced payout" if l.get("reduced_payout") else "full payout",
    "minutes risk": lambda lg, l: f"{l.get('minutes_risk') or 'unrated'} minutes risk",
}
WEIGHTS = {"type": 0.8, "cushion": 0.5, "research": 0.4, "payout": 0.3, "minutes risk": 0.5}
FEATURE_PRIOR = 8
# Leaving early costs an Over (fewer minutes to get there); PrizePicks only voids a DNP.
MINUTES_RISK_PENALTY = {"More": {"medium": 3, "high": 7}, "Less": {"high": 2}}


def logit(p: float) -> float:
    p = min(max(p, 0.01), 0.99)
    return math.log(p / (1 - p))


def history() -> list[dict]:
    hist = []
    for lg in SPORT:
        hist += [h | {"league": lg} for h in load(ddir(lg) / "leans_history.json", [])]
    return hist


def model(league: str) -> dict:
    """Shrunk hit rates by side (all leagues) and by every feature bucket (this league)."""
    hist = history()
    done = [h for h in hist if h.get("result") in ("hit", "miss")]
    side = {}
    for sd in ("More", "Less"):
        rows = [h for h in done if h["side"] == sd]
        hits = sum(h["result"] == "hit" for h in rows)
        side[sd] = {"hits": hits, "n": len(rows), "rate": (hits + SIDE_PRIOR * 0.5) / (len(rows) + SIDE_PRIOR)}
    feats: dict[str, dict] = {f: {} for f in FEATURES}
    for h in done:
        if h["league"] != league:
            continue
        for f, fn in FEATURES.items():
            b = feats[f].setdefault(fn(league, h), {"hits": 0, "n": 0, "sides": []})
            b["n"] += 1
            b["hits"] += h["result"] == "hit"
            b["sides"].append(h["side"])
    for f, buckets in feats.items():
        for b in buckets.values():
            # shrink toward what this bucket's own mix of More/Less leans would hit on side records alone
            base = sum(side[sd]["rate"] for sd in b["sides"]) / len(b["sides"])
            b["base"] = base
            b["rate"] = (b["hits"] + FEATURE_PRIOR * base) / (b["n"] + FEATURE_PRIOR)
    lg_rows = [h for h in hist if h["league"] == league]
    short = sum(h.get("result") == "short" for h in lg_rows)
    return {"side": side, "features": feats, "n": len(done), "short": short, "graded_all": len(lg_rows)}


# Final blend when the player's own numbers are available (research, his data, the lean-type record)
DATA_BLEND = {"research": 0.35, "data": 0.40, "record": 0.25}
VOLATILITY_SHRINK = 0.15
NFL_LOW_YARDAGE_CAP = 62
BREAKEVEN = 54.2  # per-pick hit rate a 5 or 6-pick Flex needs at full payout


def low_line(league: str, lean: dict) -> bool:
    """A line one event can decide: NBA counting lines of 2.5 or less."""
    return league == "nba" and lean["line"] <= 2.5


def model_weight(n: int) -> float:
    """Trust the record more as it grows: 0.3 with nothing graded, up to 0.6 at 300+ results."""
    return min(0.6, 0.3 + n / 1000)


def score(league: str, lean: dict, m: dict) -> None:
    """Attach the model's confidence and the records behind it."""
    s = m["side"][lean["side"]]
    z = logit(s["rate"])
    notes = []
    for f, fn in FEATURES.items():
        bucket = fn(league, lean)
        b = m["features"][f].get(bucket)
        if not b:
            continue
        z += WEIGHTS[f] * (logit(b["rate"]) - logit(b["base"]))
        if f in ("type", "cushion") or (f == "minutes risk" and lean.get("minutes_risk")):
            notes.append(f"{bucket} {b['hits']}/{b['n']}")
    record_p = 100 / (1 + math.exp(-z))
    w = model_weight(m["n"])
    data = lean.get("data") or {}
    # the player's own numbers: count model for NBA counting stats, else his shrunk hit rate
    data_p = data.get("stat_prob", data.get("hit_rate"))
    if data_p is None:
        conf = (1 - w) * lean["confidence"] + w * record_p
    else:
        conf = DATA_BLEND["research"] * lean["confidence"] + DATA_BLEND["data"] * data_p + DATA_BLEND["record"] * record_p
    conf -= MINUTES_RISK_PENALTY[lean["side"]].get(lean.get("minutes_risk") or "", 0)
    flags = []
    if data.get("volatility") == "high":
        conf = 50 + (conf - 50) * (1 - VOLATILITY_SHRINK)  # one event decides it: trust every estimate less
        flags.append("high volatility")
    if low_line(league, lean):
        need = 60 if lean["side"] == "Less" and lean["line"] <= 0.5 else 58
        stressed = data.get("stress_prob", data_p)
        if stressed is not None and stressed < need:
            conf = min(conf, 52)
            flags.append(f"fails the one-event test ({stressed}% if his average runs 15% against it; needs {need}%)")
    if league == "nfl" and lean["side"] == "Less" and group(league, lean["stat"], lean["line"]) in ("yards", "longest play"):
        conf = min(conf, NFL_LOW_YARDAGE_CAP)
    if data_p is not None and data_p < 50:
        flags.append(f"his own numbers say {data_p}%")
    lean["flags"] = flags
    lean["model_conf"] = int(round(min(max(conf, CONF_RANGE[0]), CONF_RANGE[1])))
    lean["low_line"] = low_line(league, lean)
    lean["lean_type"] = FEATURES["type"](league, lean)
    lean["track_record"] = (" · ".join(notes) or "no record yet") + f" (all {lean['side']} {s['hits']}/{s['n']})"


# ---------------------------------------------------------------- commands
def cmd_new(league: str) -> None:
    path = ddir(league) / "board_leans.json"
    old = load(path, None)
    if old and old.get("leans"):
        date = old.get("date") or datetime.fromisoformat(old["generated_at"]).astimezone(ET).strftime("%Y%m%d")
        save(ddir(league) / f"leans_{date}.json", old)
        print(f"archived {len(old['leans'])} leans as leans_{date}.json")
    ddir(league).mkdir(parents=True, exist_ok=True)
    save(path, {"generated_at": datetime.now(timezone.utc).isoformat(), "date": today(), "league": league,
                "game": f"{league.upper()} research leans", "props": [], "leans": [], "slate_notes": []})
    print(f"new {league} leans board for {today()}")


def cmd_add(league: str, batch_path: str, label: str, notes: list[str]) -> None:
    os.environ["PPEDGE_LEAGUE"] = league
    sys.path.insert(0, str(ROOT))
    from ppedge import fetch, headshots, parse_pp  # noqa: E402  (league is read at import)

    batch = json.loads(Path(batch_path).read_text(encoding="utf-8"))
    props, _ = parse_pp.parse(load(fetch.RAW_PP, {}), lambda *a: None)
    live = {(norm(p.player), p.stat_type): p for p in props}
    photos = headshots.load(lambda *a: None, teams={l["team"] for l in batch})
    path = ddir(league) / "board_leans.json"
    doc = load(path, None)
    if doc is None or doc.get("date") != today():
        cmd_new(league)
        doc = load(path, None)
    have = {(norm(l["player"]), l["stat"], l["side"]) for l in doc["leans"]}
    m = model(league)
    added = 0
    for l in batch:
        key = (norm(l["player"]), l["stat"], l["side"])
        cur = live.get((norm(l["player"]), l["stat"]))
        if key in have:
            continue
        if cur is None:
            print("skip (not on the board):", l["player"], l["stat"], l["line"])
            continue
        if l["side"] == "Less" and not cur.under_allowed:
            print("skip (More-only line):", l["player"], l["stat"])
            continue
        l.update(current_line=cur.line, reduced_payout=bool(cur.adjusted), start=cur.start_time,
                 photo=headshots.find(photos, l["player"], l["team"]), **{"pass": label})
        l["data"] = hitrate.analyze(league, l)
        score(league, l, m)
        doc["leans"].append(l)
        have.add(key)
        added += 1
        moved = "" if cur.line == l["line"] else f"  LINE NOW {cur.line:g}"
        print(f"+ model {l['model_conf']}% (research {l['confidence']}%) {l['player']} {l['side']} "
              f"{l['line']:g} {l['stat']} [{l['track_record']}]{moved}")
    pick_favorites(doc)
    doc["slate_notes"] += notes
    doc["researched_at"] = datetime.now(ET).strftime("%I:%M %p ET").lstrip("0")
    doc["model"] = model_summary(league, m)
    save(path, doc)
    print(f"added {added}; total {len(doc['leans'])}")


def model_summary(league: str, m: dict) -> dict:
    rows = lambda f: sorted(([k, f"{v['hits']}/{v['n']}", round(v["rate"] * 100)]  # noqa: E731
                             for k, v in m["features"][f].items()), key=lambda r: -r[2])
    return {"graded": m["n"], "left_early": m["short"],
            "side": {k: f"{v['hits']}/{v['n']}" for k, v in m["side"].items()},
            "types": rows("type"), "cushion": rows("cushion"), "research": rows("research"),
            "payout": rows("payout"), "minutes_risk": rows("minutes risk")}


def rescore(league: str, refresh_data: bool = False) -> None:
    """Re-score today's leans with the current record and re-pick the favorites."""
    path = ddir(league) / "board_leans.json"
    doc = load(path, None)
    if not doc or not doc.get("leans"):
        return
    m = model(league)
    for l in doc["leans"]:
        if refresh_data or "data" not in l:
            l["data"] = hitrate.analyze(league, l)
        score(league, l, m)
    pick_favorites(doc)
    doc["model"] = model_summary(league, m)
    save(path, doc)


MAX_LOW_LINE_FAVORITES = 2  # one-event lines are volatile: keep most favorites on bigger numbers


def pick_favorites(doc: dict) -> None:
    """The six best by model confidence (full payout breaks ties), never two from one player."""
    for l in doc["leans"]:
        l.pop("favorite", None)
    seen = set()
    n = low = 0
    for l in sorted((l for l in doc["leans"] if not l.get("cut")),
                    key=lambda l: (-l["model_conf"], l["reduced_payout"], -l["confidence"])):
        if n == 6:
            break
        if norm(l["player"]) in seen or (l.get("low_line") and low >= MAX_LOW_LINE_FAVORITES):
            continue
        seen.add(norm(l["player"]))
        low += bool(l.get("low_line"))
        n += 1
        l["favorite"] = n


# ---------------------------------------------------------------- grading
def get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return json.load(urllib.request.urlopen(req, timeout=30))


def nba_row(stats: dict) -> dict:
    """ESPN NBA box row -> PrizePicks stat names."""
    fgm, fga = map(int, stats["FG"].split("-"))
    tpm, tpa = map(int, stats["3PT"].split("-"))
    ftm, fta = map(int, stats["FT"].split("-"))
    n = lambda k: int(stats.get(k) or 0)  # noqa: E731
    pts, reb, ast = n("PTS"), n("REB"), n("AST")
    return {
        "Points": pts, "Rebounds": reb, "Assists": ast, "Steals": n("STL"), "Blocked Shots": n("BLK"),
        "Turnovers": n("TO"), "Offensive Rebounds": n("OREB"), "Defensive Rebounds": n("DREB"),
        "FG Made": fgm, "FG Attempted": fga, "3-PT Made": tpm, "3-PT Attempted": tpa,
        "Two Pointers Made": fgm - tpm, "Two Pointers Attempted": fga - tpa,
        "Free Throws Made": ftm, "Free Throws Attempted": fta, "Personal Fouls": n("PF"),
        "Pts+Rebs": pts + reb, "Pts+Asts": pts + ast, "Rebs+Asts": reb + ast, "Pts+Rebs+Asts": pts + reb + ast,
        "Blks+Stls": n("BLK") + n("STL"), "minutes": n("MIN"),
    }


def nfl_rows(summ: dict) -> dict:
    """ESPN NFL box score -> {player: {PrizePicks stat: value}}."""
    out: dict[str, dict] = {}
    num = lambda v: float(str(v).split("/")[0]) if str(v).replace(".", "").replace("/", "").isdigit() else 0.0  # noqa: E731
    for team in summ["boxscore"].get("players", []):
        for block in team["statistics"]:
            kind, labels = block["name"], block.get("labels") or []
            for a in block["athletes"]:
                r = out.setdefault(norm(a["athlete"]["displayName"]), {})
                row = dict(zip(labels, a.get("stats") or []))
                if kind == "passing":
                    comp, att = (row.get("C/ATT") or "0/0").split("/")
                    r.update({"Pass Completions": int(comp), "Pass Attempts": int(att), "Pass Yards": num(row.get("YDS")),
                              "Pass TDs": num(row.get("TD")), "INT": num(row.get("INT"))})
                elif kind == "rushing":
                    r.update({"Rush Attempts": num(row.get("CAR")), "Rush Yards": num(row.get("YDS")),
                              "Rush TDs": num(row.get("TD")), "Longest Rush": num(row.get("LONG"))})
                elif kind == "receiving":
                    r.update({"Receptions": num(row.get("REC")), "Receiving Yards": num(row.get("YDS")),
                              "Rec TDs": num(row.get("TD")), "Longest Reception": num(row.get("LONG")),
                              "Rec Targets": num(row.get("TGTS"))})
                elif kind == "kicking":
                    r.update({"FG Made": num((row.get("FG") or "0/0").split("/")[0]), "Kicking Points": num(row.get("PTS"))})
                elif kind == "punting":
                    r.update({"Punts": num(row.get("NO"))})
                elif kind == "defensive":
                    r.update({"Tackles+Ast": num(row.get("TOT")), "Sacks": num(row.get("SACKS"))})
    for r in out.values():
        r["Rush+Rec Yds"] = r.get("Rush Yards", 0) + r.get("Receiving Yards", 0)
        r["Pass+Rush Yds"] = r.get("Pass Yards", 0) + r.get("Rush Yards", 0)
        r["Anytime TDs"] = r.get("Rush TDs", 0) + r.get("Rec TDs", 0)
        if r.get("Pass Attempts"):
            r["Completion Percentage"] = round(100 * r["Pass Completions"] / r["Pass Attempts"], 1)
    return out


def box_scores(league: str, date: str) -> dict:
    base = f"https://site.api.espn.com/apis/site/v2/sports/{SPORT[league]}"
    out = {}
    for ev in get(f"{base}/scoreboard?dates={date}")["events"]:
        if ev["status"]["type"]["name"] != "STATUS_FINAL":
            continue
        summ = get(f"{base}/summary?event={ev['id']}")
        if league == "nfl":
            out.update(nfl_rows(summ))
            continue
        for team in summ["boxscore"]["players"]:
            for block in team["statistics"]:
                keys = block.get("labels") or block.get("keys")
                for a in block["athletes"]:
                    nm = norm(a["athlete"]["displayName"])
                    out[nm] = nba_row(dict(zip(keys, a["stats"]))) if a.get("stats") else {"dnp": True}
                    out[nm].update(id=a["athlete"]["id"], event=ev["id"])
    return out


def usual_minutes(league: str, athlete_id: str, event_id: str) -> float | None:
    """Average minutes in the player's other games this season (None if fewer than 2)."""
    try:
        log = get(f"https://site.web.api.espn.com/apis/common/v3/sports/{SPORT[league]}/athletes/{athlete_id}/gamelog")
        i = log["labels"].index("MIN")
        mins = [float(e["stats"][i]) for c in log["seasonTypes"][0]["categories"] for e in c.get("events", [])
                if e["eventId"] != event_id and float(e["stats"][i]) > 0]
    except (OSError, ValueError, KeyError, IndexError):
        return None
    return sum(mins) / len(mins) if len(mins) >= 2 else None


def short_reason(league: str, row: dict) -> str | None:
    """A lean doesn't count if the player left early (injury, ejection, foul-out or a blowout benching):
    under SHORT_SHARE of his usual minutes. NBA only; NFL box scores have no snap counts."""
    if league != "nba" or not row.get("minutes") or not row.get("id"):
        return None
    usual = usual_minutes(league, row["id"], row["event"])
    if usual and row["minutes"] < SHORT_SHARE * usual:
        return f"played {row['minutes']} min, usually {usual:.0f}"
    return None


def cmd_grade(league: str, date: str) -> None:
    src = ddir(league) / f"leans_{date}.json"
    if not src.exists():
        src = ddir(league) / "board_leans.json"
    leans = load(src, {}).get("leans", [])
    box = box_scores(league, date)
    graded = []
    for l in leans:
        # only leans that were on the PrizePicks board at start time count
        if l.get("current_line") is None or l.get("cut"):
            continue
        line = l["current_line"]
        row = box.get(norm(l["player"]))
        g = {k: l.get(k) for k in ("player", "team", "opp", "stat", "side", "confidence", "model_conf", "favorite",
                                   "reduced_payout", "photo", "why", "pass", "projection", "minutes_risk",
                                   "expected_minutes")}
        g["line"] = line
        act = None if row is None or row.get("dnp") else row.get(l["stat"])
        g["actual"] = act
        g["minutes"] = (row or {}).get("minutes")
        g["result"] = ("void" if act is None else "push" if act == line
                       else "hit" if (act > line) == (l["side"] == "More") else "miss")
        why_short = short_reason(league, row) if row and g["result"] in ("hit", "miss") else None
        if why_short:
            # left early: shown with what would have happened, but kept out of the record and the model
            g.update(would_have=g["result"], result="short", short_note=why_short)
        graded.append(g)
    rec = lambda rows: f"{sum(r['result'] == 'hit' for r in rows)}/{sum(r['result'] in ('hit', 'miss') for r in rows)}"  # noqa: E731
    doc = {"date": date, "league": league, "record": rec(graded),
           "favorites_record": rec([g for g in graded if g.get("favorite")]),
           "pushes": sum(g["result"] == "push" for g in graded), "voids": sum(g["result"] == "void" for g in graded),
           "short": sum(g["result"] == "short" for g in graded), "leans": graded}
    save_day(league, date, doc)
    print(f"{league} {date}: {doc['record']} (favorites {doc['favorites_record']}), "
          f"{doc['pushes']} push, {doc['voids']} void, {doc['short']} left early")
    for g in sorted(graded, key=lambda g: (g["result"], -g["confidence"])):
        print(f"{g['result']:5} {g['confidence']}% {g['player']} {g['side']} {g['line']:g} {g['stat']} -> {g['actual']}"
              f"{'  (' + g['short_note'] + ')' if g.get('short_note') else ''}")


def save_day(league: str, date: str, doc: dict) -> None:
    """Write a graded day, replace that day in the history and rebuild the results board."""
    save(ddir(league) / f"leans_results_{date}.json", doc)
    hist_path = ddir(league) / "leans_history.json"
    hist = [h for h in load(hist_path, []) if h.get("date") != date] + [{**g, "date": date} for g in doc["leans"]]
    save(hist_path, hist)
    write_results_board(league)
    rescore(league)


def cmd_void(league: str, date: str, player: str, note: str) -> None:
    """Mark a player's leans that day as left early by hand (e.g. an NFL injury the box score can't show)."""
    doc = load(ddir(league) / f"leans_results_{date}.json", None)
    if doc is None:
        sys.exit(f"no graded results for {league} {date}")
    hit = [g for g in doc["leans"] if norm(g["player"]) == norm(player)]
    for g in hit:
        if g["result"] != "short":
            g.update(would_have=g["result"], result="short", short_note=note or "left early")
    rec = lambda rows: f"{sum(r['result'] == 'hit' for r in rows)}/{sum(r['result'] in ('hit', 'miss') for r in rows)}"  # noqa: E731
    doc.update(record=rec(doc["leans"]), favorites_record=rec([g for g in doc["leans"] if g.get("favorite")]),
               short=sum(g["result"] == "short" for g in doc["leans"]))
    save_day(league, date, doc)
    print(f"{len(hit)} lean(s) for {player} marked left early; record now {doc['record']}")


def write_results_board(league: str) -> None:
    """board_results.json: every graded day (newest first) plus what the model has learned."""
    days = sorted((load(p, {}) for p in ddir(league).glob("leans_results_*.json")),
                  key=lambda d: d.get("date", ""), reverse=True)
    hist = load(ddir(league) / "leans_history.json", [])
    done = [h for h in hist if h.get("result") in ("hit", "miss")]
    save(ddir(league) / "board_results.json", {
        "generated_at": datetime.now(timezone.utc).isoformat(), "league": league,
        "game": f"{league.upper()} lean results", "props": [], "days": days,
        "overall": f"{sum(h['result'] == 'hit' for h in done)}/{len(done)}",
        "model": model_summary(league, model(league)),
    })


def cmd_vet(league: str, vet_path: str) -> None:
    """Apply a second research pass: minutes risk, late news, and keep / downgrade / drop verdicts."""
    path = ddir(league) / "board_leans.json"
    doc = load(path, None)
    vets = {(norm(v["player"]), v["stat"], v["side"]): v
            for v in json.loads(Path(vet_path).read_text(encoding="utf-8")) if v.get("league", league) == league}
    for l in doc["leans"]:
        v = vets.get((norm(l["player"]), l["stat"], l["side"]))
        if not v:
            continue
        l["minutes_risk"] = (v.get("minutes_risk") or "").lower() or None
        l["expected_minutes"] = v.get("expected_minutes")
        l["vet_note"] = " ".join(x for x in (v.get("risk_reason"), v.get("news"), v.get("note")) if x)
        verdict = (v.get("verdict") or "keep").lower()
        if verdict == "drop":
            l["cut"] = True
        elif verdict == "downgrade" and v.get("new_confidence"):
            l["confidence"] = int(v["new_confidence"])
        print(f"{verdict:9} {l['player']} {l['side']} {l['line']:g} {l['stat']} (minutes risk {l['minutes_risk']})")
    save(path, doc)
    rescore(league)


# Standing research rules. The record-based lessons are added from the history each time.
RESEARCH_RULES = """- Project every lean from per-minute (or per-snap) rate x expected minutes tonight, then adjust for
  injuries and usage, pace and total, the opponent's allowance for that stat, refs, rest and the spread.
- The season-long rate (last season and this season) must agree with recent form. Drop the lean if not.
- Rate minutes risk (low / medium / high) for every lean: injury tags, recent in-game injuries, minutes
  limits, back-to-backs, foul trouble, blowout risk, rotation changes. A More with high minutes risk is
  rarely worth it; PrizePicks only voids a player who does not play at all.
- Give a projection for every lean so the cushion (projection vs line) can be tracked.
- Confidence 55-64 normally; 65+ only when everything agrees.
- Low lines swing on one event. A Less 0.5 needs a modeled chance of at least 60%; a Less 1.5 or 2.5
  must still be 58%+ if his average runs 15% against it. Prefer bigger lines with real cushion.
- NFL Unders on low yardage or longest-play lines are capped at 62%: one long play decides them.
- A player's own per-minute numbers over this season and last (minutes-weighted, small samples shrunk
  toward the league) outweigh a hot 3-5 game stretch."""


def brief(league: str) -> str:
    """What the record says, written as instructions for the next research pass."""
    m = model(league)
    sm = model_summary(league, m)
    lines = [f"Track record so far ({sm['graded']} graded {league.upper()} leans; "
             f"{sm['left_early']} more were not counted because the player left early):",
             f"- More {sm['side']['More']}, Less {sm['side']['Less']} (all leagues)."]
    good = [t for t in sm["types"] if t[2] >= 58]
    bad = [t for t in sm["types"] if t[2] <= 47]
    if good:
        lines.append("- Winning lean types, look for more of these: " + "; ".join(f"{t[0]} {t[1]}" for t in good) + ".")
    if bad:
        lines.append("- Losing lean types, avoid unless the case is exceptional: "
                     + "; ".join(f"{t[0]} {t[1]}" for t in bad) + ".")
    if sm["cushion"]:
        lines.append("- By cushion: " + "; ".join(f"{t[0]} {t[1]}" for t in sm["cushion"]) + ".")
    if sm["research"]:
        lines.append("- By research confidence: " + "; ".join(f"{t[0]} {t[1]}" for t in sm["research"])
                     + ". Calibrate your confidence to these.")
    shorts = [h for h in history() if h.get("league") == league and h.get("result") == "short"]
    if shorts:
        lines.append("- Ruined by leaving early: " + "; ".join(f"{h['player']} ({h.get('short_note', '')})" for h in shorts)
                     + ". Check minutes risk on every lean.")
    return "\n".join(lines) + "\n\nRules:\n" + RESEARCH_RULES


def main() -> None:
    cmd, league = sys.argv[1], sys.argv[2]
    if cmd == "new":
        cmd_new(league)
    elif cmd == "add":
        cmd_add(league, sys.argv[3], sys.argv[4], sys.argv[5:])
    elif cmd == "grade":
        cmd_grade(league, sys.argv[3])
    elif cmd == "void":
        cmd_void(league, sys.argv[3], sys.argv[4], " ".join(sys.argv[5:]))
    elif cmd == "model":
        print(json.dumps(model_summary(league, model(league)), indent=1))
        rescore(league, refresh_data="--refresh" in sys.argv)
        write_results_board(league)
    elif cmd == "vet":
        cmd_vet(league, sys.argv[3])
    elif cmd == "brief":
        print(brief(league))


if __name__ == "__main__":
    main()
