"""PrizePicks vs sportsbook prop scan (Bucs at Cowboys, TNF, Oct 8 2026).

Usage:
    python props_edge.py                 # fetch (or reuse data < 10 min old), price, build lineups
    python props_edge.py --refresh       # force new pulls
    python props_edge.py --offline       # use saved raw files only, no requests
    python props_edge.py --format power  # optimize lineups for Power instead of Flex
    python props_edge.py --clv           # closing-line value across saved runs
    python props_edge.py --grade         # grade picks against data/results.csv

Information only. Nothing here places or automates entries.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from datetime import datetime, timezone

# Accented player names (e.g. Şengün) must not crash the Windows console
for _s in (sys.stdout, sys.stderr):
    if _s is not None and hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

# --league must be known before ppedge is imported (config reads it at import)
if "--league" in sys.argv:
    os.environ["PPEDGE_LEAGUE"] = sys.argv[sys.argv.index("--league") + 1].lower()

from ppedge import (adjust, board, config, fantasy, fetch, headshots, lineups, more_feeds, nba_combos, news,
                    parse_manual, parse_pp, parse_sgo, report, sharp_feeds, tracking)
from ppedge import names
from ppedge.dists import fit_mean
from ppedge.pricing import find_market, price

OUT_CSV = fetch.DATA_DIR.parent / "props_edge_results.csv"
OUT_MD = fetch.DATA_DIR.parent / "props_report.md"


def kalshi_median(ladder: dict[int, dict]) -> float | None:
    """Kalshi's implied line: where the "N+" Yes price crosses 50 cents.

    Contract N+ is the Over on N - 0.5; interpolate between neighbouring
    contracts on the bid/ask midpoint.
    """
    pts = sorted((n - 0.5, (v["yes_bid"] + v["yes_ask"]) / 2) for n, v in ladder.items())
    for (l1, m1), (l2, m2) in zip(pts, pts[1:]):
        if m1 >= 0.5 >= m2 and m1 != m2:
            return round(l1 + (m1 - 0.5) / (m1 - m2) * (l2 - l1), 1)
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--format", choices=["flex", "power"], default="flex")
    ap.add_argument("--devig", choices=["multiplicative", "additive", "power", "shin", "logit"],
                    default=config.DEVIG_METHOD)
    ap.add_argument("--no-manual", action="store_true", help="ignore manual_lines.json")
    ap.add_argument("--trials", type=int, default=config.SIM_TRIALS)
    ap.add_argument("--clv", action="store_true")
    ap.add_argument("--grade", action="store_true")
    ap.add_argument("--league", choices=["nfl", "nba", "cfb"], default=config.LEAGUE)
    args = ap.parse_args(argv)
    nba = config.LEAGUE == "nba"

    if args.clv:
        print("\n".join(tracking.clv_report()))
        return 0
    if args.grade:
        print("\n".join(tracking.grade()))
        return 0

    notes: list[str] = []

    def log(msg: str) -> None:
        print(msg)
        notes.append(msg)

    run_at = tracking.now()
    if True:
        # no SportsGameOdds and no hand-entered chat quotes: sharp books only, whole slate
        meta = fetch._load_meta()
        sgo = None
        pp = fetch.cached("pp", fetch.fetch_pp, log, args.refresh, args.offline)
        meta = fetch._load_meta()
        meta["pp_fetched_at"] = meta.get("pp_fetched_at")
        args.no_manual = True
    else:
        sgo, pp, meta = fetch.get_data(log, refresh=args.refresh, offline=args.offline)
    if pp is None:
        print("No PrizePicks data available; nothing to price.")
        return 1

    props, pp_diag = parse_pp.parse(pp, log)
    pp_raw = pp  # "pp" is reused as a loop variable below
    if props:
        starts = sorted({p.start_time for p in props if p.start_time})
        upcoming = [s for s in starts if datetime.fromisoformat(s) > datetime.now(timezone.utc)]
        config.FIRST_START = (upcoming or starts)[0]
        config.KICKOFF_ISO = starts[-1]  # keep refreshing until the last tip
        n_games = len({p.game_id for p in props if p.game_id})
        config.GAME_LABEL = f"{config.LEAGUE.upper()} · {n_games} game{'s' if n_games != 1 else ''} today"
    sgo_markets, sgo_diag = parse_sgo.parse(sgo, log) if sgo else ([], {"sgo": "unavailable"})
    manual = [] if args.no_manual else parse_manual.load(log)
    if manual:
        log(f"Manual chat quotes: {sum(len(m.quotes) for m in manual)} quote(s) from manual_lines.json "
            f"(as of midday; weighted down and dropped where fresher book data exists).")
    kal_raw = fetch.cached("kalshi", sharp_feeds.fetch_kalshi, log, args.refresh, args.offline)
    pin_raw = fetch.cached("pinnacle", sharp_feeds.fetch_pinnacle, log, args.refresh, args.offline)
    kalshi = sharp_feeds.parse_kalshi(kal_raw, log) if kal_raw else []
    pinnacle = sharp_feeds.parse_pinnacle(pin_raw, log) if pin_raw else []
    log(f"Sharp sources: Kalshi {len(kalshi)} player-stat market(s), Pinnacle {len(pinnacle)}.")
    extra = []
    for name, (fetcher, parser) in more_feeds.SOURCES.items():
        if config.ONLY_SOURCES and not ({name, "betrivers", "unibet"} if name == "kambi" else {name}) & config.ONLY_SOURCES:
            continue  # not used for pricing: skip the request entirely
        raw = fetch.cached(name, fetcher, log, args.refresh, args.offline)
        try:
            ms = parser(raw, log) if raw else []
        except Exception as exc:  # one broken feed must not stop the run
            log(f"{name}: parse failed ({type(exc).__name__}: {exc}); skipped.")
            ms = []
        log(f"{name}: {len(ms)} player-stat market(s).")
        extra.append(ms)
    markets = parse_manual.merge(sgo_markets, kalshi, pinnacle, *extra, manual)
    if config.ONLY_SOURCES:
        for m in markets:
            m.quotes = [q for q in m.quotes if q.book.lower() in config.ONLY_SOURCES]
        markets = [m for m in markets if m.quotes]
        log(f"Pricing from {', '.join(sorted(config.ONLY_SOURCES))} only.")
    full, short = parse_sgo.index_markets(markets)

    # P(at least one catch) per player, from receptions markets, to cap low yardage lines
    catch_p = {}
    for p in props:
        if p.spec_key in config.CATCH_CAPPED and p.player not in catch_p:
            probe = parse_pp.PPProp("probe", p.player, p.team, p.position, "Receptions", 0.5,
                                    "standard", None, "receptions")
            m = find_market(probe, full, short)
            r = price(probe, m, args.devig) if m else None
            catch_p[p.player] = r.p_over_raw if r else None

    priced, unmatched = [], []
    for p in props:
        if nba and p.spec_key is None:
            if nba_combos.key_for(p.stat_type) is None:
                unmatched.append(f"{p.player} {p.stat_type} {p.line:g} (stat not priced by books)")
                continue
            r = nba_combos.price_combo(p, full, short)
            if r:
                priced.append(r)
            else:
                unmatched.append(f"{p.player} {p.stat_type} {p.line:g} (missing component lines)")
            continue
        if p.spec_key is None and config.norm_stat(p.stat_type) == "fantasyscore":
            r = fantasy.price_fantasy(p, full, short)
            if r:
                priced.append(r)
            else:
                unmatched.append(f"{p.player} {p.stat_type} {p.line:g} (missing component lines)")
            continue
        if p.spec_key is None:
            unmatched.append(f"{p.player} {p.stat_type} {p.line:g} (stat not priced by books)")
            continue
        m = find_market(p, full, short)
        if m is None:
            unmatched.append(f"{p.player} {p.stat_type} {p.line:g}")
            continue
        cap = catch_p.get(p.player) if p.spec_key in config.CATCH_CAPPED else None
        r = price(p, m, args.devig, cap_over=cap)
        if r is not None and config.WORST_CASE_DEVIG:
            # +EV discipline: count a pick only at the most conservative fair
            # price across de-vig methods (power, multiplicative, Shin)
            alts = [x for x in (price(p, m, meth, cap_over=cap) for meth in config.DEVIG_CANDIDATES) if x]
            over_side = r.p_over >= 0.5
            pick = min if over_side else max
            worst = pick(alts + [r], key=lambda x: x.p_over)
            r.devig_range = (min(x.p_over for x in alts + [r]), max(x.p_over for x in alts + [r]))
            r.p_over, r.p_over_raw = worst.p_over, worst.p_over_raw
        if r is None:
            unmatched.append(f"{p.player} {p.stat_type} {p.line:g} (no usable quotes)")
        else:
            # each sharp source priced on its own at the PrizePicks line, to show side by side
            # what Pinnacle and Kalshi actually post, closest line to PrizePicks' (raw prices)
            r.by_source = {}
            # each source's own chance that the More wins at PrizePicks' exact line
            # (no shrinking; whole-number lines count a push as a refund)
            chance = {}
            for src in ("pinnacle", "kalshi"):
                qs = [q for q in m.quotes if q.book == src]
                one = price(p, parse_sgo.PlayerMarket(m.player, m.team, m.spec_key, m.sgo_stat, qs),
                            args.devig) if qs else None
                if one:
                    chance[src] = round(one.p_over_raw, 4)
            pin = [q for q in m.quotes if q.book == "pinnacle"]
            if pin:
                q = min(pin, key=lambda q: (abs(q.line - p.line), q.is_alt))
                r.by_source["pinnacle"] = {"line": q.line, "over": q.over, "under": q.under}
            # Kalshi: the full posted ladder; prefer the contract matching PrizePicks exactly
            ladder = sharp_feeds.KALSHI_LADDER.get((names.norm_name(p.player), p.spec_key)) or {}
            if ladder:
                want = math.floor(p.line) + 1  # More on 6.5 or on 6 (push at 6) needs 7+
                n = want if want in ladder else min(ladder, key=lambda k: abs(k - want))
                r.by_source["kalshi"] = {"line": n - 0.5, "n": n, **ladder[n],
                                         "exact": n == want, "median": kalshi_median(ladder)}
            for src, c in chance.items():
                if src in r.by_source:
                    r.by_source[src]["p_more"] = c
            priced.append(r)

    adj_log = adjust.apply(priced, adjust.load()) if config.USE_ADJUSTMENTS else []
    if not priced:
        print("No props matched a sportsbook price.")
        print("\n".join(unmatched))
        return 1

    # tonight's injury / lineup news: flag everyone, keep OUT and DOUBTFUL out of lineups
    tonight = news.load()
    for pp in priced:
        pp.news = news.status(tonight, pp.prop.player)
    blowout = {t for t, items in news.situations().items() if any(i["flag"] == "BLOWOUT" for i in items)}
    playable = [pp for pp in priced if not (pp.news and pp.news.get("status") in news.AVOID)
                # a minutes restriction caps Overs; Unders stay playable
                and not (pp.news and pp.news.get("status") == "MINUTES_LIMIT" and pp.pick == "Over")
                # blowout risk (spread 10+): starters may sit late, so no Overs in lineups
                and not (pp.prop.team in blowout and pp.pick == "Over")]
    if tonight:
        log(f"News: {len(tonight)} player update(s); {len(priced) - len(playable)} prop(s) held out of lineups.")

    top, extras = report.top_picks(playable)
    lu = lineups.build(playable, fmt=args.format, trials=args.trials)
    menu = lineups.build_menu(playable, trials=min(args.trials, 100_000))
    portfolio = lineups.build_portfolio(playable, trials=min(args.trials, 100_000))

    print()
    print("\n".join(report.table(priced)))
    print("\n* = at or above the 6-pick Flex breakeven. 'old' = original normal-curve method, for comparison.\n")
    print("\n".join(report.lineup_lines(lu)))
    print(f"\nNo sportsbook match ({len(unmatched)}): " + "; ".join(unmatched[:40]))

    report.write_csv(priced, OUT_CSV)
    OUT_MD.write_text(report.markdown(priced, top, extras, lu, unmatched, adj_log, meta,
                                      {"prizepicks": pp_diag, "sportsgameodds": sgo_diag}, notes),
                      encoding="utf-8")
    photos = headshots.load(log, teams={p.prop.team for p in priced if p.prop.team})
    # Combo lines vs Pinnacle sums: Pts+Rebs, Pts+Asts, Rebs+Asts (and PRA) on PrizePicks compared
    # with the sum of Pinnacle's own points / rebounds / assists lines for that player
    if nba:
        parts_for = {"pts+rebs": ("pts", "reb"), "pts+asts": ("pts", "ast"), "rebs+asts": ("reb", "ast"),
                     "pts+rebs+asts": ("pts", "reb", "ast")}
        for pp in priced:
            parts = parts_for.get(config.norm_stat(pp.prop.stat_type))
            if not parts:
                continue
            comp = []
            for key in parts:
                probe = parse_pp.PPProp("probe", pp.prop.player, pp.prop.team, None, key, 0.5, "standard", None, key)
                mk = find_market(probe, full, short)
                pins = [q for q in (mk.quotes if mk else []) if q.book == "pinnacle" and not q.is_alt]
                if not pins:
                    break
                q = pins[0]
                comp.append({"stat": key, "line": q.line, "over": q.over, "under": q.under})
            if len(comp) == len(parts):
                if not hasattr(pp, "by_source") or pp.by_source is None:
                    pp.by_source = {}
                pp.by_source["pin_sum"] = {"line": round(sum(c["line"] for c in comp), 1), "parts": comp}

    # Line-move alerts: sharp price moved since the last run while the app line didn't
    prev = news.previous_probs(board.BOARD)
    sits = news.situations()
    moves = 0
    for pp in priced:
        pp.situations = sits.get(pp.prop.team, [])
        old = prev.get((pp.prop.player, pp.prop.stat_type))
        pp.moved = None
        if old and old[0] == pp.prop.line:
            p_over_old = old[2] if old[1] == "More" else 1 - old[2]
            delta = pp.p_adj - p_over_old  # change in P(over) at the same app line
            if abs(delta) >= config.MOVE_ALERT:
                pp.moved = {"delta_over": round(delta, 4), "since": old[3]}
                moves += 1
    if moves:
        log(f"Line moves: {moves} prop(s) where the books moved {config.MOVE_ALERT:.0%}+ since the last run "
            f"but the app line did not.")
    board.write(priced, top, extras, lu, unmatched, adj_log, meta, notes, run_at, menu,
                photos=photos, portfolio=portfolio)

    # Plays board: every PrizePicks More (standard, reduced, demon, goblin) judged on one rule:
    # fair chance from Pinnacle (Kalshi as a half-weight second opinion) x payout factor vs breakeven
    plays_props, _ = parse_pp.parse(pp_raw, lambda *a: None, keep_types=("standard", "demon", "goblin"))
    # PrizePicks' own center for each player/stat = its standard line (treated as its 50% point)
    std_line = {(names.norm_name(p.player), p.spec_key): p.line
                for p in plays_props if p.odds_type == "standard" and p.spec_key}
    legs = []
    for p in plays_props:
        st = news.status(tonight, p.player)
        if p.spec_key is None or (st and st.get("status") in news.AVOID):
            continue
        m = find_market(p, full, short)
        if m is None:
            continue
        ch = {}
        for src in ("pinnacle", "kalshi"):
            qs = [q for q in m.quotes if q.book == src]
            one = price(p, parse_sgo.PlayerMarket(m.player, m.team, m.spec_key, m.sgo_stat, qs),
                        args.devig) if qs else None
            if one:
                ch[src] = one.p_over_raw
        if "pinnacle" not in ch and "kalshi" not in ch:
            continue
        w = {"pinnacle": 2.0, "kalshi": 1.0}
        fair = sum(ch[s] * w[s] for s in ch) / sum(w[s] for s in ch)
        pin = [q for q in m.quotes if q.book == "pinnacle"]
        pq = min(pin, key=lambda q: (abs(q.line - p.line), q.is_alt)) if pin else None
        lad = sharp_feeds.KALSHI_LADDER.get((names.norm_name(p.player), p.spec_key)) or {}
        want = math.floor(p.line) + 1
        # demon/goblin payout estimate: PrizePicks scales an alt line by its own odds of it,
        # centred on its standard line (50%). Pays ~ 0.5 / P_pp(over alt line).
        est = None
        center = std_line.get((names.norm_name(p.player), p.spec_key))
        if p.odds_type in ("demon", "goblin") and center is not None:
            spec = config.STATS[[s.key for s in config.STATS].index(p.spec_key)]
            dist = fit_mean(spec.family, config.spec_disp(spec, config.spec_params(spec, p.position)), center, 0.5)
            q_alt = dist.p_over_given_no_push(p.line)
            est = round(min(max(0.5 / max(q_alt, 0.01), 0.2), 4.0), 3)
        if p.odds_type in ("demon", "goblin") and est is None:
            continue  # no standard line to anchor the payout estimate
        legs.append({
            "id": p.pp_id, "player": p.player, "team": p.team, "opp": p.opponent, "game": p.game_id,
            "start": p.start_time, "stat": p.stat_type, "line": p.line,
            "type": p.odds_type if p.odds_type != "standard" else ("reduced" if p.adjusted else "standard"),
            "fair": round(fair, 4), "pin_p": round(ch["pinnacle"], 4) if "pinnacle" in ch else None,
            "kal_p": round(ch["kalshi"], 4) if "kalshi" in ch else None,
            "pin": {"line": pq.line, "over": pq.over} if pq else None,
            "kal": ({"n": want, **lad[want]} if want in lad else None),
            "photo": headshots.find(photos, p.player, p.team),
            "news": st, "est_factor": est, "pp_center": center,
        })
    doc = {"generated_at": tracking.now().isoformat(), "kickoff": config.FIRST_START or config.KICKOFF_ISO,
           "league": config.LEAGUE, "game": config.GAME_LABEL, "unit": config.UNIT, "bankroll": config.BANKROLL,
           "props": [], "legs": legs, "flex": config.FLEX_PAYOUTS, "power": config.POWER_PAYOUTS,
           "reduced_factor": config.ADJUSTED_LEG_FACTOR}
    (board.BOARD.parent / "board_plays.json").write_text(json.dumps(doc, default=str), encoding="utf-8")
    log(f"Plays board: {len(legs)} More leg(s) across standard, reduced, demon and goblin lines.")

    # Research leans: re-check every lean against the live PrizePicks board on each refresh
    leans_path = board.BOARD.parent / "board_leans.json"
    if leans_path.exists():
        try:
            ld = json.loads(leans_path.read_text(encoding="utf-8"))
            live = {(names.norm_name(p.player), p.stat_type): p for p in plays_props if p.odds_type == "standard"}
            # sharp no-vig chance the More wins at PrizePicks' exact line, for closing line value (CLV)
            sharp = {}
            for pp in priced:
                bs = getattr(pp, "by_source", None) or {}
                pin, kal = bs.get("pinnacle") or {}, bs.get("kalshi") or {}
                pm = pin.get("p_more") if pin.get("line") == pp.prop.line else None
                if pm is None and kal.get("exact"):
                    pm = kal.get("p_more")
                if pm is not None:
                    sharp[(names.norm_name(pp.prop.player), pp.prop.stat_type, pp.prop.line)] = pm
            now = tracking.now()
            for l in ld.get("leans", []):
                cur = live.get((names.norm_name(l["player"]), l["stat"]))
                if cur is None and l.get("start") and datetime.fromisoformat(l["start"]) <= now:
                    if not l.get("locked") and l.get("market_now") is not None:
                        l["market_close"] = l["market_now"]  # last sharp price before the game = the close
                    l["locked"] = True  # game started: PrizePicks drops the line, keep the last one
                    continue
                if cur and l.get("current_line") not in (None, cur.line):
                    # every PrizePicks move is logged: pulls and moves show where informed money went
                    l.setdefault("line_moves", []).append({"at": now.isoformat(), "from": l["current_line"], "to": cur.line})
                pm = sharp.get((names.norm_name(l["player"]), l["stat"], cur.line)) if cur else None
                if pm is not None:
                    l["market_now"] = round(100 * (pm if l["side"] == "More" else 1 - pm), 1)
                    l.setdefault("market_open", l["market_now"])
                l["current_line"] = cur.line if cur else None
                l["reduced_payout"] = bool(cur and cur.adjusted)
                if cur:
                    l["start"] = cur.start_time
                l["news"] = news.status(tonight, l["player"])
            ld["checked_at"] = tracking.now().isoformat()
            leans_path.write_text(json.dumps(ld, indent=1, default=str), encoding="utf-8")
        except (OSError, ValueError, KeyError) as exc:
            log(f"Leans re-check skipped ({type(exc).__name__}).")

    # Mores-only board: every playable line judged on its More side alone
    # only lines Kalshi or Pinnacle price directly (no Fantasy Score / built combos)
    # and only full-payout standard lines: reduced-payout ("adjusted") lines cut the multiplier
    # every line Pinnacle prices (full and reduced payout, labeled on the card)
    mores = copy.deepcopy([pp for pp in playable if (getattr(pp, "by_source", None) or {}).get("pinnacle")])
    for pp in mores:
        pp.prop.allowed = "over"  # forces pick = More; probability is P(over) from the books
    board.write(mores, [], [], [], unmatched, [], meta, notes, run_at,
                lineups.build_menu(mores, trials=min(args.trials, 100_000)), photos=photos,
                portfolio=lineups.build_portfolio(mores, trials=min(args.trials, 100_000)),
                path=board.BOARD.parent / "board_mores.json", mode="mores")
    log(f"Mores board: {sum(1 for pp in mores if pp.prob > 0.5)} More pick(s) above 50%.")

    # Narrative Board: same book prices with narratives applied on top (NBA: tonight's news).
    narr = copy.deepcopy(playable)
    narr_entries = adjust.load() + (news.to_narratives(tonight) if nba else [])
    narr_log = adjust.apply_narratives(narr, narr_entries)
    board.write(narr, [], [], [], unmatched, narr_log, meta, notes, run_at,
                lineups.build_menu(narr, trials=min(args.trials, 100_000)), photos=photos,
                portfolio=lineups.build_portfolio(narr, trials=min(args.trials, 100_000)),
                path=board.BOARD_NARR, mode="narrative")
    tracking.log_run(priced, run_at)
    snap = fetch.snapshot([fetch.RAW_SGO, fetch.RAW_PP, fetch.RAW_META, OUT_CSV, OUT_MD, board.BOARD,
                           fetch.DATA_DIR / "raw_kalshi.json", fetch.DATA_DIR / "raw_pinnacle.json"])
    print(f"\nSaved {OUT_CSV.name}, {OUT_MD.name}; snapshot in {snap.relative_to(fetch.DATA_DIR.parent)}")
    print(f"Data pulled: SportsGameOdds {report.fmt_time(meta.get('sgo_fetched_at'))}; "
          f"PrizePicks {report.fmt_time(meta.get('pp_fetched_at'))}. Recheck lines in the app before entering.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
