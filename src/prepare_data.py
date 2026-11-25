"""Step 2: load, clean and split the EESA corpus, then print dataset statistics.

Usage:
    python src/prepare_data.py --input data/raw/eesa.csv

GUESSES (check against the real file):
  - the file is a CSV with a text column and a label column
  - the column names are "text" and "label" (change with --text-col / --label-col)
"""
import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42  # fixed seed so every run gives the same split


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
ARABIC_RE = re.compile(r"[؀-ۿ]")
LATIN_RE = re.compile(r"[A-Za-z]")
DIACRITICS_RE = re.compile(r"[ً-ْـ]")  # tashkeel + tatweel


def clean_text(text: str) -> str:
    """Light cleaning only: we must not delete words that carry sentiment."""
    text = str(text)
    text = URL_RE.sub(" ", text)          # links carry no sentiment
    text = MENTION_RE.sub(" ", text)      # usernames are personal data / noise
    text = DIACRITICS_RE.sub("", text)    # same word written two ways -> one way
    text = re.sub(r"\s+", " ", text).strip()
    return text


def script_type(text: str) -> str:
    """Rough code-switching tag: Arabic letters only, Latin only, or both."""
    has_ar = bool(ARABIC_RE.search(text))
    has_la = bool(LATIN_RE.search(text))
    if has_ar and has_la:
        return "mixed"
    if has_ar:
        return "arabic"
    if has_la:
        return "latin"
    return "other"


def stats(df: pd.DataFrame, text_col: str, label_col: str) -> dict:
    words = df[text_col].str.split().str.len()
    return {
        "rows": int(len(df)),
        "labels": df[label_col].value_counts().to_dict(),
        "script": df["script"].value_counts().to_dict(),
        "words_mean": round(float(words.mean()), 2),
        "words_max": int(words.max()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out-dir", default="data/processed")
    p.add_argument("--text-col", default="text")
    p.add_argument("--label-col", default="label")
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()
    set_seed(args.seed)

    # 1. Load
    path = Path(args.input)
    df = pd.read_excel(path) if path.suffix in {".xlsx", ".xls"} else pd.read_csv(path)
    df = df[[args.text_col, args.label_col]].rename(
        columns={args.text_col: "text", args.label_col: "label"})
    n_raw = len(df)

    # 2. Clean
    df = df.dropna()
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.len() > 0]
    n_before_dedup = len(df)
    df = df.drop_duplicates(subset="text")  # duplicates could leak train -> test
    df["script"] = df["text"].map(script_type)
    df = df.reset_index(drop=True)

    # 3. Split 80/10/10, stratified so each split has the same label mix
    train, rest = train_test_split(df, test_size=0.2, stratify=df["label"],
                                   random_state=args.seed)
    dev, test = train_test_split(rest, test_size=0.5, stratify=rest["label"],
                                 random_state=args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("dev", dev), ("test", test)]:
        part.to_csv(out / f"{name}.csv", index=False)

    # 4. Statistics (needed for the report)
    report = {
        "seed": args.seed,
        "raw_rows": n_raw,
        "removed_empty_or_missing": n_raw - n_before_dedup,
        "removed_duplicates": n_before_dedup - len(df),
        "all": stats(df, "text", "label"),
        "train": stats(train, "text", "label"),
        "dev": stats(dev, "text", "label"),
        "test": stats(test, "text", "label"),
    }
    (out / "stats.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
