"""LLM self-explanations + erasure faithfulness (ERASER comprehensiveness / sufficiency).

For each sentence the LLM (1) predicts a label, (2) lists the words that drove its decision.
Faithfulness is measured with the SAME LLM's probability for its predicted label:
  comprehensiveness = p(full text) - p(text WITHOUT rationale words)   (higher = more faithful)
  sufficiency       = p(full text) - p(ONLY rationale words)           (lower  = more faithful)
A random word set of the same size is the control.
Usage: python src/explain_llm.py --model Qwen/Qwen2.5-7B-Instruct --n 50"""
import argparse
import random
import re

import pandas as pd
import torch

from common import (SEED, set_seed, label_names, save_result, RES, load_eval_items, norm_word,
                    rationale_set, match_indices, remove_words, keep_words, token_f1, mean)
from llm_common import load_llm, chat_prompt, LabelScorer

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
p.add_argument("--n", type=int, default=50)
p.add_argument("--no-4bit", action="store_true")
args = p.parse_args()
set_seed()
rng = random.Random(SEED)

labels = label_names()
items = load_eval_items(args.n)
tok, model = load_llm(args.model, not args.no_4bit)
scorer = LabelScorer(tok, model, labels)


@torch.no_grad()
def self_explain(text: str, label: str) -> list:
    msg = (f"Text:\n{text}\n\nThe sentiment of this text is {label}. List the words that most "
           f"influenced this sentiment. Copy each word exactly as written in the text, separated "
           f"by commas, at most 5 words. Output only the words.")
    ids = tok(chat_prompt(tok, msg), return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(**ids, max_new_tokens=40, do_sample=False)
    ans = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
    cand = [c.strip() for c in re.split(r"[,،\n;]", ans) if c.strip()]
    text_norm = {norm_word(w) for w in text.split()}
    return [c for c in cand if any(norm_word(x) in text_norm for x in c.split())][:5]


COLS = ["id", "pred", "gold_label", "explanation", "p_full", "comprehensiveness", "sufficiency",
        "comp_random", "suff_random", "human_token_f1"]
rows = []
n_total = len(items)
for _, it in items.iterrows():
    words = it["text"].split()
    pr = scorer.probs(it["text"])
    k = max(range(len(labels)), key=lambda i: pr[i])
    expl = self_explain(it["text"], labels[k])
    idx = match_indices(words, rationale_set(";".join(expl)))
    if not idx:
        continue                                   # LLM gave no usable words
    ridx = rng.sample(range(len(words)), min(len(idx), len(words)))   # random control
    p_full = pr[k]
    p_rm = scorer.probs(remove_words(words, idx))[k]
    p_only = scorer.probs(keep_words(words, idx))[k]
    p_rm_r = scorer.probs(remove_words(words, ridx))[k]
    p_only_r = scorer.probs(keep_words(words, ridx))[k]
    gold = match_indices(words, rationale_set(it["rationale_words"])) if it["rationale_words"] else []
    rows.append({"id": it["id"], "pred": labels[k], "gold_label": it["label"],
                 "explanation": ";".join(expl), "p_full": p_full,
                 "comprehensiveness": p_full - p_rm, "sufficiency": p_full - p_only,
                 "comp_random": p_full - p_rm_r, "suff_random": p_full - p_only_r,
                 "human_token_f1": token_f1(set(idx), set(gold)) if gold else None})

df = pd.DataFrame(rows, columns=COLS)
RES.mkdir(exist_ok=True)
tag = args.model.split("/")[-1]
df.to_csv(RES / f"explain_llm_{tag}_items.csv", index=False)
save_result(f"explain_llm_{tag}", {
    "model": args.model, "n_items_total": n_total, "n_items_with_usable_explanation": len(df), "human_rationales": bool(items["rationale_words"].str.len().any()),
    "comprehensiveness": mean(df["comprehensiveness"]), "comprehensiveness_random": mean(df["comp_random"]),
    "sufficiency": mean(df["sufficiency"]), "sufficiency_random": mean(df["suff_random"]),
    "human_token_f1": mean(df["human_token_f1"].dropna())})
