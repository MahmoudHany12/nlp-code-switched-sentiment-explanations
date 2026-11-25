"""First Integrated Gradients (Sundararajan et al., 2017) on a fine-tuned encoder.
IG for input x with baseline x0 and target class c:
    IG_i = (x_i - x0_i) * mean over alpha in (0,1] of  dF_c(x0 + alpha (x - x0)) / dx_i
computed on the input EMBEDDINGS (so it works for text). Token scores are summed to words.
Faithfulness uses the same erasure scores as explain_llm.py, on the encoder's own probability.
Usage: python src/ig_encoder.py --model-dir models/MARBERTv2 --n 50 --steps 32"""
import argparse
import random

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import (SEED, set_seed, save_result, RES, load_eval_items, rationale_set, match_indices,
                    remove_words, keep_words, token_f1, mean)

p = argparse.ArgumentParser()
p.add_argument("--model-dir", required=True)
p.add_argument("--n", type=int, default=50)
p.add_argument("--steps", type=int, default=32)
p.add_argument("--chunk", type=int, default=16, help="IG steps per forward pass (memory)")
p.add_argument("--topk", type=int, default=3, help="words kept when no human rationale exists")
args = p.parse_args()
set_seed()
rng = random.Random(SEED)
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(args.model_dir)
model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(dev).eval()
for prm in model.parameters():
    prm.requires_grad_(False)
labels = [model.config.id2label[i] for i in range(model.config.num_labels)]


def encode(text):
    return tok(text if text.strip() else tok.unk_token, truncation=True, max_length=128,
               return_tensors="pt").to(dev)


@torch.no_grad()
def probs(text):
    return torch.softmax(model(**encode(text)).logits[0], -1)


def integrated_gradients(text, target):
    enc = encode(text)
    emb_layer = model.get_input_embeddings()
    x = emb_layer(enc["input_ids"]).detach()                      # (1, T, H)
    base_ids = enc["input_ids"].clone()
    special = torch.tensor(tok.get_special_tokens_mask(base_ids[0].tolist(),
                                                       already_has_special_tokens=True), device=dev).bool()
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    base_ids[0, ~special] = pad                                   # baseline: all words -> [PAD]
    x0 = emb_layer(base_ids).detach()
    total = torch.zeros_like(x)
    alphas = torch.linspace(1.0 / args.steps, 1.0, args.steps, device=dev)
    for i in range(0, args.steps, args.chunk):
        a = alphas[i:i + args.chunk].view(-1, 1, 1)
        z = (x0 + a * (x - x0)).requires_grad_(True)
        mask = enc["attention_mask"].expand(len(a), -1)
        out = model(inputs_embeds=z, attention_mask=mask).logits[:, target].sum()
        total += torch.autograd.grad(out, z)[0].sum(0, keepdim=True)
    attr = ((x - x0) * total / args.steps).sum(-1)[0]            # (T,) token attributions
    with torch.no_grad():                                         # completeness check
        delta = (model(inputs_embeds=x, attention_mask=enc["attention_mask"]).logits[0, target]
                 - model(inputs_embeds=x0, attention_mask=enc["attention_mask"]).logits[0, target])
    return enc, attr.detach(), float(delta - attr.sum())


def word_scores(enc, attr, n_words):
    scores = [0.0] * n_words
    for t, w in enumerate(enc.word_ids(0)):
        if w is not None and w < n_words:
            scores[w] += float(attr[t])
    return scores


items = load_eval_items(args.n)
rows = []
for _, it in items.iterrows():
    words = it["text"].split()
    if not words:
        continue
    pr = probs(it["text"])
    k = int(pr.argmax())
    enc, attr, err = integrated_gradients(it["text"], k)
    sc = word_scores(enc, attr, len(words))
    gold = match_indices(words, rationale_set(it["rationale_words"])) if it["rationale_words"] else []
    kk = min(len(gold) if gold else args.topk, len(words))
    top = sorted(range(len(words)), key=lambda i: -sc[i])[:kk]
    rnd = rng.sample(range(len(words)), kk)
    pf = float(pr[k])
    rows.append({"id": it["id"], "pred": labels[k], "gold_label": it["label"],
                 "ig_top_words": ";".join(words[i] for i in top), "p_full": pf,
                 "completeness_error": err,
                 "comprehensiveness": pf - float(probs(remove_words(words, top))[k]),
                 "sufficiency": pf - float(probs(keep_words(words, top))[k]),
                 "comp_random": pf - float(probs(remove_words(words, rnd))[k]),
                 "suff_random": pf - float(probs(keep_words(words, rnd))[k]),
                 "human_token_f1": token_f1(set(top), set(gold)) if gold else None})

df = pd.DataFrame(rows)
RES.mkdir(exist_ok=True)
tag = args.model_dir.rstrip("/").split("/")[-1]
df.to_csv(RES / f"ig_{tag}_items.csv", index=False)
save_result(f"ig_{tag}", {
    "model_dir": args.model_dir, "n_items": len(df), "ig_steps": args.steps,
    "human_rationales": bool(items["rationale_words"].str.len().any()),
    "mean_abs_completeness_error": mean(df["completeness_error"].abs()),
    "comprehensiveness": mean(df["comprehensiveness"]), "comprehensiveness_random": mean(df["comp_random"]),
    "sufficiency": mean(df["sufficiency"]), "sufficiency_random": mean(df["suff_random"]),
    "human_token_f1": mean(df["human_token_f1"].dropna())})
