"""Player-name and team normalization for cross-source matching."""

from __future__ import annotations

import re
import unicodedata

from . import config

_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def norm_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
    s = s.lower().replace(".", "").replace("'", "").replace("-", " ")
    s = _SUFFIX.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


def short_key(name: str) -> str:
    """First initial + last name, e.g. 'c lamb'. Used as a fallback key."""
    parts = norm_name(name).split()
    if len(parts) < 2:
        return norm_name(name)
    return f"{parts[0][0]} {parts[-1]}"


def norm_team(value: str | None) -> str | None:
    if not value:
        return None
    v = str(value).upper().replace("_NFL", "").replace("_", " ").strip()
    for abbr, aliases in config.TEAM_ALIASES.items():
        if v in aliases or any(a in v for a in aliases if len(a) > 3):
            return abbr
    return v
