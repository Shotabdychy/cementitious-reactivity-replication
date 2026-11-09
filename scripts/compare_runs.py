"""Compare a re-run against the saved first run, ignoring wall-clock fields.

usage: python compare_runs.py <baseline> <new> [--only-variant base]
Prints IDENTICAL or the first differing values. Exit code 0 either way; the verdict is the text.
"""
import os
import sys
import json

import pandas as pd

IGNORE = {"secs", "wall_seconds", "elapsed", "python", "tensorflow", "pandas"}


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k not in IGNORE}
    if isinstance(o, list):
        return [strip(v) for v in o]
    return o


def diff_json(a, b, path=""):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            out += diff_json(a.get(k), b.get(k), f"{path}.{k}" if path else k)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} -> {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                out += diff_json(x, y, f"{path}[{i}]")
    else:
        if isinstance(a, float) and isinstance(b, float):
            if not (a == b or (a != a and b != b)):
                out.append(f"{path}: {a} -> {b}")
        elif a != b:
            out.append(f"{path}: {a!r} -> {b!r}")
    return out


base, new = sys.argv[1], sys.argv[2]
only_variant = None
if "--only-variant" in sys.argv:
    only_variant = sys.argv[sys.argv.index("--only-variant") + 1]

label = os.path.basename(new)
if not os.path.exists(base):
    print(f"  {label}: no baseline")
    sys.exit(0)
if not os.path.exists(new):
    print(f"  {label}: MISSING after re-run")
    sys.exit(0)

if new.endswith(".json"):
    d = diff_json(strip(json.load(open(base))), strip(json.load(open(new))))
    print(f"  {label}: {'IDENTICAL' if not d else 'DIFFERS'}")
    for line in d[:12]:
        print(f"      {line}")
else:
    a, b = pd.read_csv(base), pd.read_csv(new)
    if only_variant and "variant" in a.columns:
        a = a[a.variant == only_variant].reset_index(drop=True)
        b = b[b.variant == only_variant].reset_index(drop=True)
        label += f" (variant '{only_variant}' only)"
    same = a.shape == b.shape and a.astype(str).equals(b.astype(str))
    print(f"  {label}: {'IDENTICAL' if same else 'DIFFERS'}")
    if not same:
        if a.shape != b.shape:
            print(f"      shape {a.shape} -> {b.shape}")
        else:
            ne = (a.astype(str) != b.astype(str))
            for r, c in list(zip(*ne.values.nonzero()))[:12]:
                print(f"      row {r} col {a.columns[c]}: {a.iat[r, c]} -> {b.iat[r, c]}")
