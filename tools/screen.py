"""Screen every PrizePicks line on today's board with the player-data model, both sides.

    python tools/screen.py <league> [out.json]

Research then verifies the best of these (minutes, usage, injuries) instead of hunting for leans by feel:
on the first graded day the data model sorted hits from misses far better than research confidence did.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import hitrate  # noqa: E402

MIN_PROB = 60  # report candidates the data model gives at least this chance


def main() -> None:
    league = sys.argv[1]
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data" / league / "screen.json"
    os.environ["PPEDGE_LEAGUE"] = league
    sys.path.insert(0, str(ROOT))
    from ppedge import fetch, headshots, parse_pp  # noqa: E402  (league is read at import)

    props, _ = parse_pp.parse(json.loads(fetch.RAW_PP.read_text(encoding="utf-8")), lambda *a: None)
    props = [p for p in props if "fantasy" not in p.stat_type.lower()]
    photos = headshots.load(lambda *a: None, teams={p.team for p in props if p.team})
    found = []
    for i, p in enumerate(props):
        base = {"player": p.player, "team": p.team, "opp": p.opponent, "stat": p.stat_type, "line": p.line,
                "current_line": p.line, "photo": headshots.find(photos, p.player, p.team),
                "reduced_payout": bool(p.adjusted), "start": p.start_time}
        for side in ("More", "Less") if p.under_allowed else ("More",):
            lean = dict(base, side=side)
            try:
                d = hitrate.analyze(league, lean)
            except Exception as exc:  # one bad log must not stop the screen
                print(f"skip {p.player} {p.stat_type}: {type(exc).__name__}", file=sys.stderr)
                d = None
            if not d:
                continue
            prob = d.get("stat_prob", d.get("hit_rate"))
            if prob is not None and prob >= MIN_PROB:
                found.append(dict(lean, data=d, data_prob=prob))
        if i % 50 == 0:
            print(f"{i}/{len(props)} lines screened", file=sys.stderr)
    found.sort(key=lambda c: -c["data_prob"])
    out.write_text(json.dumps(found, indent=1), encoding="utf-8")
    print(f"{len(found)} candidate(s) at {MIN_PROB}%+ written to {out}")
    for c in found[:40]:
        d = c["data"]
        print(f"{c['data_prob']}% (stress {d.get('stress_prob')}) {c['player']} {c['side']} {c['line']:g} {c['stat']}"
              f" | avg {d['avg']} over {d['games']} games, proj {d.get('proj')}{' | reduced' if c['reduced_payout'] else ''}")


if __name__ == "__main__":
    main()
