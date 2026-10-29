"""Build PrizePicks Flex slips from the leans boards: positive-edge legs only, highest expected value first.

    python tools/slips.py [size] [count]

Rules (VIX correlation study, 2025-26 logs): different players are close to independent; two legs on
the same player are one minutes bet, so one leg per player; one Clingan-style center pair counts once.
Each slip's EV uses the Flex payout table times each leg's payout factor (reduced lines pay ~12% less).
Stake: quarter Kelly on the slip's edge, capped at 1.5% of bankroll.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ppedge import config  # noqa: E402

KELLY = 0.25
MAX_STAKE = 0.015  # of bankroll per slip
MAX_POOL = 12  # best legs considered (keeps the search small)


def legs() -> list[dict]:
    out = []
    for lg in ("nba", "nfl"):
        doc = json.loads((ROOT / "data" / lg / "board_leans.json").read_text(encoding="utf-8"))
        for l in doc["leans"]:
            if l.get("cut") or l.get("current_line") is None or l.get("edge", 0) <= 0:
                continue
            out.append({"league": lg, "player": l["player"], "team": l.get("team"), "stat": l["stat"],
                        "side": l["side"], "line": l["current_line"], "p": l["model_conf"] / 100,
                        "factor": config.ADJUSTED_LEG_FACTOR if l.get("reduced_payout") else 1.0,
                        "edge": l["edge"]})
    return sorted(out, key=lambda x: -x["edge"])[:MAX_POOL]


def flex_ev(ps: list[float], factors: list[float], n: int) -> tuple[float, list[float]]:
    dist = [1.0]
    for p in ps:
        nxt = [0.0] * (len(dist) + 1)
        for h, x in enumerate(dist):
            nxt[h + 1] += x * p
            nxt[h] += x * (1 - p)
        dist = nxt
    scale = 1.0
    for f in factors:
        scale *= f
    table = config.FLEX_PAYOUTS[n]
    return sum(pr * table.get(h, 0) * scale for h, pr in enumerate(dist)), dist


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    pool = legs()
    slips = []
    for combo in itertools.combinations(pool, n):
        if len({c["player"] for c in combo}) < n:
            continue  # one leg per player
        ev, _ = flex_ev([c["p"] for c in combo], [c["factor"] for c in combo], n)
        slips.append((ev, combo))
    slips.sort(key=lambda s: -s[0])
    used, picked = set(), []
    for ev, combo in slips:  # diversify: each new slip shares at most 3 legs with earlier ones
        keys = {(c["player"], c["stat"]) for c in combo}
        if all(len(keys & k) <= 3 for k in used):
            picked.append((ev, combo))
            used.add(frozenset(keys))
        if len(picked) == count:
            break
    for i, (ev, combo) in enumerate(picked, 1):
        # Kelly for a multi-outcome payout, approximated by the slip's edge over its variance-free odds
        edge = ev - 1
        stake = min(max(edge, 0) * KELLY, MAX_STAKE) * config.BANKROLL
        print(f"Slip {i}: {n}-pick Flex, EV {edge:+.1%}, suggested ${stake:.2f}")
        for c in combo:
            print(f"   {c['league'].upper()} {c['player']} {c['side']} {c['line']:g} {c['stat']}"
                  f"  ({c['p']:.0%}{', reduced' if c['factor'] < 1 else ''})")


if __name__ == "__main__":
    main()
