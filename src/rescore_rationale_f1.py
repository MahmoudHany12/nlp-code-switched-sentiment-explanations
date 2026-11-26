"""Recompute word-level F1 between saved explanations and the current rationales.

Use after editing data/rationales/rationales.csv: no model is re-run.
- LLM: the saved explanation words are matched to the sentence exactly as in explain_llm.py.
- IG: the saved top words are mapped back to their positions in the sentence.
  Note: IG top-k was chosen with k = length of the rationale at run time (the AI draft),
  so IG comprehensiveness/sufficiency are unchanged; re-run ig_encoder.py to use new lengths.
Usage: python src/rescore_rationale_f1.py"""
import json
import pandas as pd
from common import RES, RATIONALES, match_indices, rationale_set, token_f1, mean

r = pd.read_csv(RATIONALES, dtype=str).fillna("")
gold = dict(zip(r["id"], r["rationale_words"]))
text = dict(zip(r["id"], r["text"]))


def ig_positions(words, top_words):
    used, out = set(), []
    for tw in top_words.split(";"):
        for i, w in enumerate(words):
            if i not in used and w == tw:
                used.add(i); out.append(i); break
    return out


for path in sorted(RES.glob("*_items.csv")):
    df = pd.read_csv(path, dtype={"id": str})
    is_ig = "ig_top_words" in df.columns
    f1 = []
    for _, x in df.iterrows():
        words = text[x["id"]].split()
        g = match_indices(words, rationale_set(gold[x["id"]])) if gold.get(x["id"]) else []
        if is_ig:
            p = ig_positions(words, str(x["ig_top_words"]))
        else:
            p = match_indices(words, rationale_set(str(x["explanation"])))
        f1.append(token_f1(set(p), set(g)) if g else None)
    df["human_token_f1"] = f1
    df.to_csv(path, index=False)
    js = path.with_name(path.name.replace("_items.csv", ".json"))
    res = json.loads(js.read_text())
    res["human_token_f1"] = mean(df["human_token_f1"].dropna())
    res["rationales_version"] = "reviewed (v2)"
    js.write_text(json.dumps(res, indent=2))
    print(path.name, res["human_token_f1"])
