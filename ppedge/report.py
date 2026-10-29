"""Console, CSV and Markdown output."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from . import config
from .lineups import LineupResult, banned_pair, breakeven
from .names import norm_name
from .pricing import PricedProp

ET = ZoneInfo("America/New_York")


def _side(pp: PricedProp) -> str:
    return "More" if pp.pick == "Over" else "Less"


def _reason(pp: PricedProp) -> str:
    bits = [f"{'SHARP' if pp.tier == 'sharp' else 'retail only'}: books {pp.book_line:g} vs PP {pp.prop.line:g}"]
    if pp.method == "ladder":
        bits.append("fit to multiple lines")
    if pp.line_gap_sd > 0.25:
        bits.append(f"line gap {pp.line_gap_sd:.2f} sd (shrunk)")
    if pp.p_push > 0.01:
        bits.append(f"push {pp.p_push:.0%}")
    adj = [f"{k} {d * 100:+.1f}" for k, d in pp.adjustments if abs(d) > 1e-9]
    if adj:
        bits.append("adj: " + ", ".join(adj))
    if not pp.prop.under_allowed:
        bits.append("More only")
    return "; ".join(bits)


def top_picks(priced: list[PricedProp], n: int = 10, extra: int = 2):
    ranked = sorted((pp for pp in priced if pp.tier == "sharp" or not config.LINEUPS_SHARP_ONLY),
                    key=lambda pp: -pp.prob)
    top, seen = [], set()
    for pp in ranked:
        k = norm_name(pp.prop.player)
        if k not in seen:
            top.append(pp)
            seen.add(k)
        if len(top) == n:
            break
    rest = [pp for pp in ranked if pp not in top][:extra]
    return top, rest


def table(priced: list[PricedProp]) -> list[str]:
    hdr = (f"{'#':>3} {'player':22} {'tm':3} {'stat':18} {'PP':>6} {'pick':4} {'fair':>6} "
           f"{'odds':>6} {'bk':>3} {'adj':>6} {'old':>6}  reason")
    out = [hdr, "-" * len(hdr)]
    for i, pp in enumerate(sorted(priced, key=lambda x: -x.prob), 1):
        flag = " *" if pp.prob >= config.FLAG_THRESHOLD else "  "
        old = pp.p_over_normal if pp.pick == "Over" else 1 - pp.p_over_normal
        out.append(
            f"{i:>3} {pp.prop.player[:22]:22} {pp.prop.team or '':3} {pp.prop.stat_type[:18]:18} "
            f"{pp.prop.line:>6g} {_side(pp):4} {pp.fair_prob:>6.1%} {pp.fair_american:>+6.0f} "
            f"{pp.n_books:>3} {pp.prob:>6.1%}{flag}{old:>6.1%}  {_reason(pp)}")
    return out


def lineup_lines(results: list[LineupResult]) -> list[str]:
    out = []
    for i, r in enumerate(results):
        name = "AB"[i] if i < 2 else str(i + 1)
        indep = f", {r.ev_independent:.3f} if legs were independent" if r.ev_independent else ""
        out.append(f"Lineup {name} ({len(r.legs)}-pick {r.fmt.title()}): expected return {r.ev:.3f} per $1{indep}, "
                   f"P(profit) {r.p_profit:.1%}, P(all hit) {r.p_all:.1%}")
        for pp in r.legs:
            out.append(f"   {pp.prop.player:22} {_side(pp):4} {pp.prop.line:g} {pp.prop.stat_type:18} {pp.prob:.1%}")
    return out


def write_csv(priced: list[PricedProp], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["player", "team", "position", "stat", "pp_line", "pick", "fair_prob", "fair_american",
                    "n_books", "books", "book_line", "method", "p_push", "reliability", "raw_market_prob",
                    "adjusted_prob", "old_normal_prob", "clears_54.2", "reason"])
        for pp in sorted(priced, key=lambda x: -x.prob):
            raw = pp.p_over_raw if pp.pick == "Over" else 1 - pp.p_over_raw
            old = pp.p_over_normal if pp.pick == "Over" else 1 - pp.p_over_normal
            w.writerow([pp.prop.player, pp.prop.team, pp.prop.position, pp.prop.stat_type, pp.prop.line,
                        _side(pp), round(pp.fair_prob, 4), round(pp.fair_american), pp.n_books,
                        " ".join(pp.books), pp.book_line, pp.method, round(pp.p_push, 4),
                        round(pp.reliability, 3), round(raw, 4), round(pp.prob, 4), round(old, 4),
                        pp.prob >= config.FLAG_THRESHOLD, _reason(pp)])


def fmt_time(iso: str | None) -> str:
    if not iso:
        return "not pulled"
    return datetime.fromisoformat(iso).astimezone(ET).strftime("%I:%M %p ET, %b %d")


def markdown(priced, top, extras, lineups, unmatched, adj_log, meta, diag, notes) -> str:
    L = ["# PrizePicks vs Sportsbooks: Bucs at Cowboys", ""]
    L.append(f"- SportsGameOdds pulled: {fmt_time(meta.get('sgo_fetched_at'))}")
    L.append(f"- PrizePicks pulled: {fmt_time(meta.get('pp_fetched_at'))}")
    L.append(f"- Breakeven per leg: 6-pick Flex {breakeven('flex', 6):.1%}, 2-pick Power {breakeven('power', 2):.1%}, "
             f"6-pick Power {breakeven('power', 6):.1%} (payout tables in ppedge/config.py; verify in app)")
    L += [f"- {n}" for n in notes]
    L += ["", "## Top 10 (one per player)", "",
          "| # | Player | Stat | Line | Pick | Fair | Adjusted | Books | Why |", "|---|---|---|---|---|---|---|---|---|"]
    for i, pp in enumerate(top, 1):
        L.append(f"| {i} | {pp.prop.player} ({pp.prop.team}) | {pp.prop.stat_type} | {pp.prop.line:g} | {_side(pp)} | "
                 f"{pp.fair_prob:.1%} ({pp.fair_american:+.0f}) | {pp.prob:.1%} | {pp.n_books} | {_reason(pp)} |")
    L += ["", "## Two extra props", ""]
    for pp in extras:
        L.append(f"- {pp.prop.player} {_side(pp)} {pp.prop.line:g} {pp.prop.stat_type}: {pp.prob:.1%} ({_reason(pp)})")
    L += ["", "## Lineups", ""] + [f"    {l}" for l in lineup_lines(lineups)]
    L += ["", "## Situational adjustments", ""]
    for a in adj_log:
        status = "priced in, 0 applied" if a["priced_in"] else f"{a['applied_pp']:+.1f} pts applied"
        who = ", ".join(a["matched"]) or "no matching PrizePicks prop"
        L.append(f"- **{a['id']}** ({status}): {a['reason']} Matched: {who}.")
    L += ["", "## No sportsbook match", ""]
    L += [f"- {u}" for u in unmatched] or ["- none"]
    L += ["", "## Full board", "", "```"] + table(priced) + ["```", "", "## Diagnostics", "", "```", str(diag), "```"]
    return "\n".join(L)
