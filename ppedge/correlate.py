"""Pairwise correlation model between prop outcomes (in Over/Over terms).

Values are judgment-based priors reflecting common NFL relationships:
same-player stat overlap, QB-to-receiver stacking, target competition,
backfield committees, and game script (Dallas favored, Bucs trailing).
"""

from __future__ import annotations

import numpy as np

from . import config
from .names import norm_name
from .pricing import PricedProp

FAVORITE = "DAL"

PASS = {"pass_yds", "pass_att", "pass_comp", "pass_td", "pass_rush_yds", "long_comp", "pass_int", "sacks_taken"}
RUSH = {"rush_yds", "rush_att", "long_rush", "rush_td"}
REC = {"rec_yds", "receptions", "targets", "long_rec", "rec_td"}
MIXED = {"rush_rec_yds", "tds"}
DEF = {"tackles", "sacks"}
KICK = {"kick_pts", "fg_made"}
PUNT = {"punts", "punts_in20"}

SAME_FAMILY = [
    {"rec_yds", "receptions", "targets", "long_rec", "rush_rec_yds"},
    {"rush_yds", "rush_att", "long_rush", "rush_rec_yds", "pass_rush_yds"},
    {"pass_yds", "pass_att", "pass_comp", "long_comp", "pass_rush_yds"},
]


def _role(pp: PricedProp) -> str:
    k = pp.prop.spec_key
    pos = (pp.prop.position or "").upper()
    if k in PASS:
        return "pass"
    if k in RUSH:
        return "qbrush" if pos == "QB" else "rush"
    if k in REC:
        return "rbrec" if pos == "RB" else "rec"
    if k == "rush_rec_yds":
        return "rush" if pos == "RB" else "rec"
    if k in DEF:
        return "def"
    if k in KICK:
        return "kick"
    if k in PUNT:
        return "punt"
    return "other"


NBA_PARTS = {"pts": {"pts"}, "reb": {"reb"}, "ast": {"ast"}, "fg3m": {"pts", "fg3m"},
             "pra": {"pts", "reb", "ast"}, "stl": {"stl"}, "blk": {"blk"}}


def _nba_parts(pp: PricedProp) -> set:
    if pp.prop.spec_key in NBA_PARTS:
        return NBA_PARTS[pp.prop.spec_key]
    s = pp.prop.stat_type.lower()
    return {k for k, w in (("pts", "pts"), ("reb", "reb"), ("ast", "ast"), ("blk", "blk"), ("stl", "stl"))
            if w in s} or ({"pts", "reb", "ast"} if "fantasy" in s else set())


def nba_rho(a: PricedProp, b: PricedProp) -> float:
    """NBA: legs in different games are independent; same-team usage competes."""
    pa, pb = a.prop, b.prop
    if pa.game_id and pb.game_id and pa.game_id != pb.game_id:
        return 0.0
    A, B = _nba_parts(a), _nba_parts(b)
    if norm_name(pa.player) == norm_name(pb.player):
        return 0.6 if A & B else 0.15
    if pa.team == pb.team:
        if "pts" in A and "pts" in B:
            return -0.10  # shots and usage are shared
        if ("ast" in A and "pts" in B) or ("ast" in B and "pts" in A):
            return 0.10  # assists feed teammates' points
        if "reb" in A and "reb" in B:
            return -0.10
        return 0.0
    if "pts" in A and "pts" in B:
        return 0.08  # same game, opposing players: pace and total
    return 0.0


def pair_rho(a: PricedProp, b: PricedProp) -> float:
    if config.LEAGUE == "nba":
        return nba_rho(a, b)
    pa, pb = a.prop, b.prop
    ka, kb = pa.spec_key, pb.spec_key
    ra, rb = _role(a), _role(b)
    if norm_name(pa.player) == norm_name(pb.player):
        if any(ka in f and kb in f for f in SAME_FAMILY):
            return 0.65
        if {ra, rb} == {"pass", "qbrush"}:
            return -0.10
        if {ra, rb} == {"rush", "rbrec"}:
            return -0.15
        return 0.30
    roles = {ra, rb}
    if pa.team == pb.team:
        if roles in ({"pass", "rec"}, {"pass", "rbrec"}):
            return 0.35 if "rec" in roles else 0.20
        if roles == {"pass", "rush"}:
            return -0.15
        if ra == rb == "rec":
            return -0.10
        if ra == rb == "rbrec":
            return -0.25
        if ra == rb == "rush":
            return -0.25
        if roles == {"rbrec", "rec"}:
            return -0.08
        if "kick" in roles and roles & {"pass", "rec", "rush", "qbrush"}:
            return 0.15
        if "punt" in roles and roles & {"pass", "rec", "rush", "qbrush"}:
            return -0.20
        return 0.0
    # opponents
    fav = a if pa.team == FAVORITE else b
    dog = b if fav is a else a
    rf, rd = _role(fav), _role(dog)
    # cross-team links are weak in game logs; kept small so they cannot dominate EV
    if rf in {"pass", "rec"} and rd in {"pass", "rec"}:
        return 0.07  # shootout / game total
    if rd in {"pass", "rec", "rbrec"} and rf == "rush":
        return 0.06  # Bucs trailing and throwing, Dallas running late
    if rd == "rush" and rf in {"pass", "rec"}:
        return 0.03
    if "def" in roles:
        other = rb if ra == "def" else ra
        return 0.20 if other in {"rush", "pass", "rec", "qbrush"} else 0.0
    return 0.0


def sign(pp: PricedProp) -> float:
    return 1.0 if pp.pick == "Over" else -1.0


def matrix(legs: list[PricedProp]) -> np.ndarray:
    n = len(legs)
    c = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            rho = pair_rho(legs[i], legs[j]) * sign(legs[i]) * sign(legs[j])
            c[i, j] = c[j, i] = rho
    return nearest_psd(c)


def nearest_psd(c: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(c)
    if w.min() > 1e-8:
        return c
    w = np.clip(w, 1e-6, None)
    c2 = v @ np.diag(w) @ v.T
    d = np.sqrt(np.diag(c2))
    return c2 / np.outer(d, d)
