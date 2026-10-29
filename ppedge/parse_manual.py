"""Load hand-entered quotes (manual_lines.json) as extra sportsbook markets."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from . import config
from .devig import american_to_prob
from .names import norm_name
from .parse_sgo import PlayerMarket, Quote

MANUAL_FILE = Path(__file__).resolve().parent.parent / "manual_lines.json"
SPEC_BY_KEY = {s.key: s for s in config.STATS}


def load(log, path: Path = MANUAL_FILE) -> list[PlayerMarket]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not doc.get("enabled", True):
        return []
    as_of = datetime.fromisoformat(doc["as_of"]) if doc.get("as_of") else None
    grouped: dict[tuple, PlayerMarket] = {}
    for q in doc.get("quotes", []):
        spec = SPEC_BY_KEY.get(q.get("stat"))
        if spec is None:
            log(f"manual_lines.json: unknown stat '{q.get('stat')}', skipped.")
            continue
        line = float(q["line"])
        book = "manual:" + q.get("book", "unknown")
        if q.get("prob") is not None:
            quote = Quote(book, line, 0, 0, as_of, False, "prob", float(q["prob"]))
        elif q.get("over") is not None and q.get("under") is not None:
            quote = Quote(book, line, float(q["over"]), float(q["under"]), as_of, False)
        elif q.get("over") is not None:
            p = american_to_prob(q["over"]) / (1.0 + config.ONE_SIDED_HOLD)
            quote = Quote(book, line, float(q["over"]), 0, as_of, False, "one_sided", p)
        elif q.get("under") is not None:
            p = 1.0 - american_to_prob(q["under"]) / (1.0 + config.ONE_SIDED_HOLD)
            quote = Quote(book, line, 0, float(q["under"]), as_of, False, "one_sided", p)
        else:
            quote = Quote(book, line, 0, 0, as_of, False, "line_only", 0.5)
        key = (norm_name(q["player"]), q.get("team"), spec.key)
        if key not in grouped:
            grouped[key] = PlayerMarket(q["player"], q.get("team"), spec.key, spec.sgo_ids[0], [])
        grouped[key].quotes.append(quote)
    return list(grouped.values())


def merge(*market_lists: list[PlayerMarket]) -> list[PlayerMarket]:
    """Combine markets for the same player/stat from different sources."""
    out: dict[tuple, PlayerMarket] = {}
    for markets in market_lists:
        for m in markets:
            key = (norm_name(m.player), m.team, m.spec_key)
            if key in out:
                out[key].quotes.extend(m.quotes)
            else:
                out[key] = PlayerMarket(m.player, m.team, m.spec_key, m.sgo_stat, list(m.quotes))
    return list(out.values())
