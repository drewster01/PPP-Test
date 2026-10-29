"""Pick logging, grading, calibration and closing-line value (CLV)."""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .fetch import DATA_DIR
from .names import norm_name
from .pricing import PricedProp

LOG = DATA_DIR / "picks_log.csv"
RESULTS = DATA_DIR / "results.csv"
FIELDS = ["run_at", "pp_id", "player", "team", "stat", "pp_line", "pick", "p_market",
          "p_adj", "p_normal", "book_line", "n_books", "method", "start_time"]


def log_run(priced: list[PricedProp], run_at: datetime) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    new = not LOG.exists()
    with LOG.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for pp in priced:
            w.writerow({
                "run_at": run_at.isoformat(), "pp_id": pp.prop.pp_id, "player": pp.prop.player,
                "team": pp.prop.team, "stat": pp.prop.spec_key, "pp_line": pp.prop.line,
                "pick": pp.pick, "p_market": round(pp.fair_prob, 4), "p_adj": round(pp.prob, 4),
                "p_normal": round(pp.p_over_normal if pp.pick == "Over" else 1 - pp.p_over_normal, 4),
                "book_line": pp.book_line, "n_books": pp.n_books, "method": pp.method,
                "start_time": pp.prop.start_time,
            })


def _rows() -> list[dict]:
    if not LOG.exists():
        return []
    with LOG.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def clv_report() -> list[str]:
    """Compare each prop's first logged market probability to its last pre-kickoff one."""
    by_key = defaultdict(list)
    for r in _rows():
        start = datetime.fromisoformat(r["start_time"]) if r.get("start_time") else None
        run = datetime.fromisoformat(r["run_at"])
        if start and run >= start:
            continue
        by_key[(r["player"], r["stat"], r["pp_line"], r["pick"])].append((run, float(r["p_market"])))
    lines = []
    for (player, stat, line, pick), obs in sorted(by_key.items()):
        if len(obs) < 2:
            continue
        obs.sort()
        delta = obs[-1][1] - obs[0][1]
        lines.append(f"{player:24} {stat:12} {pick:5} {line:>6}  first {obs[0][1]:.1%} -> last {obs[-1][1]:.1%}  CLV {delta:+.1%}")
    return lines or ["Need at least two runs before kickoff to measure CLV."]


def grade() -> list[str]:
    """Grade the last pre-kickoff pick per prop against data/results.csv.

    results.csv columns: player, stat, actual   (stat uses the keys in config.py)
    """
    if not RESULTS.exists():
        RESULTS.write_text("player,stat,actual\n", encoding="utf-8")
        return [f"Fill in {RESULTS} with final stats (player,stat,actual), then rerun --grade."]
    with RESULTS.open(encoding="utf-8") as f:
        actual = {(norm_name(r["player"]), r["stat"]): float(r["actual"]) for r in csv.DictReader(f) if r.get("actual")}
    last = {}
    for r in _rows():
        start = datetime.fromisoformat(r["start_time"]) if r.get("start_time") else None
        if start and datetime.fromisoformat(r["run_at"]) >= start:
            continue
        last[(r["player"], r["stat"], r["pp_line"])] = r
    graded = []
    for (player, stat, line), r in last.items():
        a = actual.get((norm_name(player), stat))
        if a is None:
            continue
        line_f = float(line)
        if a == line_f:
            continue  # push
        won = (a > line_f) == (r["pick"] == "Over")
        graded.append((float(r["p_adj"]), float(r["p_market"]), float(r["p_normal"]), won))
    if not graded:
        return ["No graded picks yet."]
    out = [f"Graded {len(graded)} picks."]
    for name, idx in (("adjusted", 0), ("market", 1), ("old normal method", 2)):
        brier = sum((g[idx] - g[3]) ** 2 for g in graded) / len(graded)
        out.append(f"Brier score ({name}): {brier:.4f}   (lower is better; 0.25 = coin flip)")
    bins = defaultdict(list)
    for g in graded:
        bins[min(int(g[0] * 20) / 20, 0.95)].append(g[3])
    out.append("Calibration (predicted -> actual hit rate):")
    for b in sorted(bins):
        v = bins[b]
        out.append(f"  {b:.0%}-{b + 0.05:.0%}: {sum(v) / len(v):.0%} of {len(v)}")
    return out


def now() -> datetime:
    return datetime.now(timezone.utc)
