"""Research leans: daily board, grading against box scores, and a model that learns from results.

    python tools/leans.py new   <league>                       # archive the old board, start today's
    python tools/leans.py add   <league> <batch.json> <label> [slate note ...]
    python tools/leans.py grade <league> <YYYYMMDD>            # grade that day's archived leans
    python tools/leans.py model <league>                       # print what the history says

Every graded lean goes into data/<league>/leans_history.json. The model turns that history into
hit rates per (stat group, side) and blends them with the researcher's confidence, so lean types
that keep losing get marked down and the ones that keep winning get marked up.
"""
from __future__ import annotations

import json
import os
import sys
import unicodedata
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

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


def model(league: str) -> dict:
    """Shrunk hit rates by side (all leagues) and by (group, side) for this league."""
    hist = []
    for lg in SPORT:
        hist += [h | {"league": lg} for h in load(ddir(lg) / "leans_history.json", [])]
    done = [h for h in hist if h.get("result") in ("hit", "miss")]
    side = {}
    for sd in ("More", "Less"):
        rows = [h for h in done if h["side"] == sd]
        hits = sum(h["result"] == "hit" for h in rows)
        side[sd] = {"hits": hits, "n": len(rows), "rate": (hits + SIDE_PRIOR * 0.5) / (len(rows) + SIDE_PRIOR)}
    groups = {}
    for h in done:
        if h["league"] != league:
            continue
        k = (group(league, h["stat"], h["line"]), h["side"])
        g = groups.setdefault(k, {"hits": 0, "n": 0})
        g["n"] += 1
        g["hits"] += h["result"] == "hit"
    for (_, sd), g in groups.items():
        g["rate"] = (g["hits"] + GROUP_PRIOR * side[sd]["rate"]) / (g["n"] + GROUP_PRIOR)
    return {"side": side, "groups": groups, "n": len(done)}


def score(league: str, lean: dict, m: dict) -> None:
    """Attach the model's confidence and the track record behind it."""
    grp = group(league, lean["stat"], lean["line"])
    g = m["groups"].get((grp, lean["side"]))
    s = m["side"][lean["side"]]
    rate = g["rate"] if g else s["rate"]
    conf = (1 - MODEL_WEIGHT) * lean["confidence"] + MODEL_WEIGHT * 100 * rate
    lean["model_conf"] = int(round(min(max(conf, CONF_RANGE[0]), CONF_RANGE[1])))
    lean["lean_type"] = f"{grp} {lean['side']}"
    rec = f"{g['hits']}/{g['n']}" if g else "no record yet"
    lean["track_record"] = f"{grp} {lean['side']}: {rec} (all {lean['side']} leans {s['hits']}/{s['n']})"


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
        score(league, l, m)
        doc["leans"].append(l)
        have.add(key)
        added += 1
        moved = "" if cur.line == l["line"] else f"  LINE NOW {cur.line:g}"
        print(f"+ model {l['model_conf']}% (research {l['confidence']}%) {l['player']} {l['side']} "
              f"{l['line']:g} {l['stat']} [{l['track_record']}]{moved}")
    # favorites: the six best by model confidence, full payout first
    ranked = sorted(doc["leans"], key=lambda l: (-l["model_conf"], l["reduced_payout"], -l["confidence"]))
    for l in doc["leans"]:
        l.pop("favorite", None)
    for i, l in enumerate(ranked[:6], 1):
        l["favorite"] = i
    doc["slate_notes"] += notes
    doc["researched_at"] = datetime.now(ET).strftime("%I:%M %p ET").lstrip("0")
    doc["model"] = model_summary(league, m)
    save(path, doc)
    print(f"added {added}; total {len(doc['leans'])}")


def model_summary(league: str, m: dict) -> dict:
    return {"graded": m["n"],
            "side": {k: f"{v['hits']}/{v['n']}" for k, v in m["side"].items()},
            "types": sorted(([f"{g} {sd}", f"{v['hits']}/{v['n']}", round(v["rate"] * 100)]
                             for (g, sd), v in m["groups"].items()), key=lambda r: -r[2])}


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
    return out


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
                                   "reduced_payout", "photo", "why", "pass", "projection")}
        g["line"] = line
        act = None if row is None or row.get("dnp") else row.get(l["stat"])
        g["actual"] = act
        g["minutes"] = (row or {}).get("minutes")
        g["result"] = ("void" if act is None else "push" if act == line
                       else "hit" if (act > line) == (l["side"] == "More") else "miss")
        graded.append(g)
    rec = lambda rows: f"{sum(r['result'] == 'hit' for r in rows)}/{sum(r['result'] in ('hit', 'miss') for r in rows)}"  # noqa: E731
    doc = {"date": date, "league": league, "record": rec(graded),
           "favorites_record": rec([g for g in graded if g.get("favorite")]),
           "pushes": sum(g["result"] == "push" for g in graded), "voids": sum(g["result"] == "void" for g in graded),
           "leans": graded}
    save(ddir(league) / f"leans_results_{date}.json", doc)
    hist_path = ddir(league) / "leans_history.json"
    hist = [h for h in load(hist_path, []) if h.get("date") != date] + [{**g, "date": date} for g in graded]
    save(hist_path, hist)
    write_results_board(league)
    print(f"{league} {date}: {doc['record']} (favorites {doc['favorites_record']}), "
          f"{doc['pushes']} push, {doc['voids']} void")
    for g in sorted(graded, key=lambda g: (g["result"], -g["confidence"])):
        print(f"{g['result']:5} {g['confidence']}% {g['player']} {g['side']} {g['line']:g} {g['stat']} -> {g['actual']}")


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


def main() -> None:
    cmd, league = sys.argv[1], sys.argv[2]
    if cmd == "new":
        cmd_new(league)
    elif cmd == "add":
        cmd_add(league, sys.argv[3], sys.argv[4], sys.argv[5:])
    elif cmd == "grade":
        cmd_grade(league, sys.argv[3])
    elif cmd == "model":
        print(json.dumps(model_summary(league, model(league)), indent=1))
        write_results_board(league)


if __name__ == "__main__":
    main()
