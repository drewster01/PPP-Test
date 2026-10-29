"""Calibration backtest for the player-data model on real past games (NBA).

    python tools/backtest.py [n_samples]

For a random sample of player-games, the model sees only games played before that night and prices a
line set where a book would put it (his recent average rounded to x.5), both sides. Scores: Brier
(lower is better; 0.25 is a coin flip) and hit rate in confidence bands. Tune on 2025-26, check on 2026-27.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hitrate  # noqa: E402
import nbastats  # noqa: E402

STATS = ["Points", "Rebounds", "Assists", "Steals", "Blocked Shots", "Turnovers", "Offensive Rebounds",
         "Defensive Rebounds", "FG Attempted", "FG Made", "3-PT Attempted", "3-PT Made",
         "Two Pointers Attempted", "Free Throws Attempted", "Pts+Rebs+Asts"]
MIN_PRIOR_GAMES = 3


def samples(season: int, prev: int, n: int, seed: int = 7) -> list[tuple]:
    """(lean, now, last, actual) where now = this season's games before that night."""
    cur, old = nbastats.game_logs(season), nbastats.game_logs(prev)
    rng = random.Random(seed)
    out = []
    names = [k for k, v in cur.items() if len(v) > MIN_PRIOR_GAMES]
    while len(out) < n and names:
        name = rng.choice(names)
        games = cur[name]
        i = rng.randrange(MIN_PRIOR_GAMES, len(games))
        g, stat = games[i], rng.choice(STATS)
        if g["minutes"] < hitrate.MIN_MINUTES:
            continue  # the model's minutes come from research for players who play
        recent = [x[stat] for x in games[max(0, i - 10):i]]
        line = round(sum(recent) / len(recent)) + rng.choice((-0.5, 0.5))
        if line < 0.5:
            continue
        lean = {"player": name, "stat": stat, "line": line, "side": "More"}
        out.append((lean, games[:i], old.get(name, []), g[stat]))
    return out


def score(rows: list[tuple]) -> dict:
    brier, n, bands = 0.0, 0, {}
    for lean, now, last, actual in rows:
        d = hitrate.model_from_games("nba", lean, now, last)
        if not d or d.get("stat_prob") is None:
            continue
        p = d["stat_prob"] / 100
        y = actual > lean["line"]
        for prob, hit in ((p, y), (1 - p, not y)):  # both sides, like real leans
            brier += (prob - hit) ** 2
            n += 1
            b = min(int(prob * 100) // 5 * 5, 95)
            hb = bands.setdefault(b, [0, 0])
            hb[0] += hit
            hb[1] += 1
    return {"brier": round(brier / max(n, 1), 4), "n": n,
            "bands": {f"{b}-{b + 5}": f"{h}/{t} ({h / t:.0%})" for b, (h, t) in sorted(bands.items()) if b >= 55}}


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    train, test = samples(2026, 2025, n), samples(2027, 2026, n // 2)
    print("2025-26 (tune):", score(train))
    print("2026-27 (check):", score(test))


if __name__ == "__main__":
    main()
