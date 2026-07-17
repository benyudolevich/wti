"""
Fit-result caching for expensive model fits (TVTP Markov-switching regime
model, joint HMM) so repeated script/chart runs don't re-fit from scratch
every time -- both fits take minutes.

Cache is keyed on a fingerprint of the TRAINING-period data actually going
into the fit (hashed directly, not file mtime) plus the relevant config
values, so it auto-invalidates if either changes -- appending new
post-TRAIN_END rows to the workbook does NOT invalidate the cache (the
frozen training fit shouldn't change from that), but editing historical
training-period values, or changing a model parameter like state count,
does.
"""

import hashlib
import pickle
from pathlib import Path

import numpy as np

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)


def hash_series(series):
    arr = np.ascontiguousarray(series.to_numpy())
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def fingerprint(*values):
    raw = "|".join(str(v) for v in values)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_or_fit(name, fp, fit_fn):
    cache_path = CACHE_DIR / f"{name}.pkl"

    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if cached.get("fingerprint") == fp:
                print(f"[cache] {name}: loaded cached fit (fingerprint {fp})")
                return cached["result"]
            print(f"[cache] {name}: fingerprint changed ({cached.get('fingerprint')} -> {fp}), refitting")
        except (pickle.PickleError, EOFError, AttributeError, ModuleNotFoundError) as e:
            print(f"[cache] {name}: cache unreadable ({e}), refitting")
    else:
        print(f"[cache] {name}: no cache found, fitting")

    result = fit_fn()
    with open(cache_path, "wb") as f:
        pickle.dump({"fingerprint": fp, "result": result}, f)
    return result