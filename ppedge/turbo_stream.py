"""Minimal decoder for React Router's "turbo-stream" page data.

Public pages built with React Router embed their loader data as a flat JSON
array in which objects reference other entries by index:
  {"_12": 34}  -> key = arr[12], value = decode(34)
  [5, 6, 7]    -> list of decoded entries
Negative integers are special values (undefined, null, NaN, ...). Tagged
arrays like ["D", ms] (date) or ["P", n] (promise) are reduced to plain values.
"""

from __future__ import annotations

import json
import re

_SPECIAL = {-1: None, -2: None, -3: float("nan"), -4: float("inf"), -5: float("-inf"), -6: 0.0, -7: None}


def extract(html: str) -> list:
    chunks = re.findall(r'streamController\.enqueue\(("(?:[^"\\]|\\.)*")\)', html)
    if not chunks:
        return []
    text = "".join(json.loads(c) for c in chunks)
    return json.loads(text.split("\n")[0])


def decode(arr: list, root: int = 0):
    memo: dict[int, object] = {}

    def dec(i):
        if isinstance(i, int) and i < 0:
            return _SPECIAL.get(i)
        if i in memo:
            return memo[i]
        v = arr[i]
        if isinstance(v, dict):
            out: dict = {}
            memo[i] = out
            for k, vi in v.items():
                key = arr[int(k[1:])] if k.startswith("_") else k
                out[key] = dec(vi)
            return out
        if isinstance(v, list):
            if v and isinstance(v[0], str) and len(v[0]) == 1 and v[0] in "DPYRESMBZ":
                memo[i] = v[1] if len(v) > 1 else None  # tagged value: keep its payload
                return memo[i]
            out_l: list = []
            memo[i] = out_l
            out_l.extend(dec(x) for x in v)
            return out_l
        memo[i] = v
        return v

    return dec(root)
