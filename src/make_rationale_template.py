"""Write data/rationales/rationales_template.csv: 50 test sentences for YOU to annotate.
For each row, fill 'rationale_words' with the words (copied exactly from the text, separated
by ';') that make you choose the label. Save the filled file as data/rationales/rationales.csv."""
import argparse
from common import SEED, RATIONALES, load_split

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=50)
n = p.parse_args().n
df = load_split("test").sample(n=n, random_state=SEED)[["id", "text", "label"]]
df["rationale_words"] = ""
out = RATIONALES.parent / "rationales_template.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False, encoding="utf-8-sig")
print(f"wrote {out} ({len(df)} rows)")
