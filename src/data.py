"""Load, clean and split a sentiment corpus; print statistics.

Sources:
  --source eesa    EESA corpus, official train/dev/test files (--eesa-dir <folder with EESA-*.csv>)
  --source arsas   public Arabic tweet sentiment corpus (STAND-IN, downloaded from Hugging Face)
  --source csv     your own file, e.g. EESA:  --input data/raw/eesa.csv --text-col X --label-col Y

Outputs data/processed/{train,dev,test}.csv, labels.json, stats.json
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from common import SEED, DATA, set_seed

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
ARABIC_RE = re.compile(r"[؀-ۿ]")
LATIN_RE = re.compile(r"[A-Za-z]")
DIACRITICS_RE = re.compile(r"[ً-ْـ]")  # tashkeel + tatweel


def clean_text(text) -> str:
    """Light cleaning only: never delete words that carry sentiment."""
    text = URL_RE.sub(" ", str(text))
    text = MENTION_RE.sub(" ", text)
    text = DIACRITICS_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def script_type(text: str) -> str:
    """Rough code-switching tag: Arabic letters only, Latin only, or both ('mixed')."""
    ar, la = bool(ARABIC_RE.search(text)), bool(LATIN_RE.search(text))
    return "mixed" if ar and la else "arabic" if ar else "latin" if la else "other"


def load_raw(args) -> pd.DataFrame:
    if args.source == "arsas":
        from datasets import load_dataset
        df = load_dataset("arbml/ArSAS", split="train").to_pandas()
        names = {0: "negative", 1: "neutral", 2: "positive", 3: "mixed"}
        df = df.rename(columns={"Tweet_text": "text"})
        df["label"] = df["label"].map(names)
        df = df[df["label"] != "mixed"]  # keep 3 clear classes
        return df[["text", "label"]]
    path = Path(args.input)
    df = pd.read_excel(path) if path.suffix in {".xlsx", ".xls"} else pd.read_csv(path)
    return df[[args.text_col, args.label_col]].rename(
        columns={args.text_col: "text", args.label_col: "label"})


def stats(df: pd.DataFrame) -> dict:
    words = df["text"].str.split().str.len()
    return {"rows": int(len(df)), "labels": df["label"].value_counts().to_dict(),
            "script": df["script"].value_counts().to_dict(),
            "words_mean": round(float(words.mean()), 2), "words_max": int(words.max())}


def main_eesa(args) -> None:
    """EESA ships official splits (CSV, NO header: comment, sentiment). We keep them, clean text,
    drop duplicates inside and across splits (priority train > dev > test) so nothing leaks."""
    set_seed(args.seed)
    parts, seen, report = {}, set(), {"source": "eesa", "seed": args.seed}
    for name, f in [("train", "EESA-Train"), ("dev", "EESA-Dev"), ("test", "EESA-Test")]:
        df = pd.read_csv(Path(args.eesa_dir) / f"{f}.csv", header=None, names=["text", "label"]).dropna()
        n_raw = len(df)
        df["text"] = df["text"].map(clean_text)
        df["label"] = df["label"].astype(str).str.strip().str.lower()
        df = df[df["text"].str.len() > 0].drop_duplicates(subset="text")
        df = df[~df["text"].isin(seen)]
        seen |= set(df["text"])
        df["script"] = df["text"].map(script_type)
        parts[name] = df.reset_index(drop=True)
        report[name + "_raw_rows"] = n_raw
    allx = pd.concat(parts.values())
    ids = iter(range(len(allx)))
    DATA.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        part.insert(0, "id", [next(ids) for _ in range(len(part))])
        part.to_csv(DATA / f"{name}.csv", index=False)
    (DATA / "labels.json").write_text(json.dumps(sorted(allx["label"].unique())))
    report.update({k: stats(v) for k, v in parts.items()})
    (DATA / "stats.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["eesa", "arsas", "csv"], default="eesa")
    p.add_argument("--input")
    p.add_argument("--eesa-dir", default="data/raw/eesa-corpus")
    p.add_argument("--text-col", default="text")
    p.add_argument("--label-col", default="label")
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()
    if args.source == "eesa":
        return main_eesa(args)
    set_seed(args.seed)

    df = load_raw(args)                                   # 1. load
    n_raw = len(df)
    df = df.dropna()                                      # 2. clean
    df["text"] = df["text"].map(clean_text)
    df["label"] = df["label"].astype(str)
    df = df[df["text"].str.len() > 0]
    n_clean = len(df)
    df = df.drop_duplicates(subset="text")                # duplicates could leak train -> test
    df["script"] = df["text"].map(script_type)
    df = df.reset_index(drop=True)
    df.insert(0, "id", range(len(df)))

    train, rest = train_test_split(df, test_size=0.2, stratify=df["label"],    # 3. split 80/10/10
                                   random_state=args.seed)
    dev, test = train_test_split(rest, test_size=0.5, stratify=rest["label"],
                                 random_state=args.seed)
    DATA.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("dev", dev), ("test", test)]:
        part.to_csv(DATA / f"{name}.csv", index=False)
    labels = sorted(df["label"].unique())
    (DATA / "labels.json").write_text(json.dumps(labels))

    report = {"source": args.source, "seed": args.seed, "raw_rows": n_raw,
              "removed_empty": n_raw - n_clean, "removed_duplicates": n_clean - len(df),
              "all": stats(df), "train": stats(train), "dev": stats(dev), "test": stats(test)}
    (DATA / "stats.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))   # 4. stats
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
