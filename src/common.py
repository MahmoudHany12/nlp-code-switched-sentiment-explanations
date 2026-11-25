"""Shared helpers: seeds, paths, data loading, metrics, word-level erasure utilities."""
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RES = ROOT / "results"
RATIONALES = ROOT / "data" / "rationales" / "rationales.csv"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{name}.csv")


def label_names() -> list:
    return json.loads((DATA / "labels.json").read_text())


def save_result(name: str, obj: dict) -> None:
    RES.mkdir(exist_ok=True)
    (RES / f"{name}.json").write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def classification_scores(y_true, y_pred) -> dict:
    from sklearn.metrics import accuracy_score, f1_score
    return {"accuracy": round(accuracy_score(y_true, y_pred), 4),
            "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4)}


# ---- word-level utilities for erasure (ERASER-style) -----------------------
def norm_word(w: str) -> str:
    """Lower-case and strip punctuation so 'جميل،' matches 'جميل'."""
    return re.sub(r"[^\w]", "", w.lower())


def rationale_set(rationale: str) -> set:
    """'good;very bad' -> {'good','very','bad'}"""
    return {norm_word(x) for part in str(rationale).split(";") for x in part.split()} - {""}


def match_indices(words: list, rset: set) -> list:
    return [i for i, w in enumerate(words) if norm_word(w) in rset]


def remove_words(words: list, idx) -> str:
    idx = set(idx)
    return " ".join(w for i, w in enumerate(words) if i not in idx)


def keep_words(words: list, idx) -> str:
    idx = set(idx)
    return " ".join(w for i, w in enumerate(words) if i in idx)


def token_f1(pred: set, gold: set) -> float:
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(gold)
    return 2 * p * r / (p + r)


def load_eval_items(n: int) -> pd.DataFrame:
    """Items for the explanation experiments.
    If data/rationales/rationales.csv has filled 'rationale_words', use those items
    (human rationales). Otherwise use n random test items and set rationale_words=''."""
    test = load_split("test")
    if RATIONALES.exists():
        r = pd.read_csv(RATIONALES, dtype=str).fillna("")
        r = r[r["rationale_words"].str.strip() != ""]
        if len(r):
            return r[["id", "text", "label", "rationale_words"]].reset_index(drop=True)
    print("WARNING: no human rationales found; using random test items without rationales.")
    items = test.sample(n=min(n, len(test)), random_state=SEED)[["id", "text", "label"]].copy()
    items["rationale_words"] = ""
    return items.reset_index(drop=True)


def mean(xs):
    xs = list(xs)
    return round(float(np.mean(xs)), 4) if xs else None
