"""Baseline type 3: zero-shot LLM classification (4-bit, fits a free Colab T4).
Usage: python src/baselines_llm.py --model Qwen/Qwen2.5-7B-Instruct --n 300"""
import argparse

from common import set_seed, load_split, label_names, save_result, classification_scores, SEED
from llm_common import load_llm, LabelScorer

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--n", type=int, default=300, help="number of test sentences (T4 time)")
p.add_argument("--no-4bit", action="store_true")
args = p.parse_args()
set_seed()

labels = label_names()
te = load_split("test").sample(n=min(args.n, len(load_split("test"))), random_state=SEED)
tok, model = load_llm(args.model, not args.no_4bit)
scorer = LabelScorer(tok, model, labels)
pred = [labels[max(range(len(labels)), key=lambda i: pr[i])]
        for pr in (scorer.probs(t) for t in te["text"])]
save_result(f"llm_{args.model.split('/')[-1]}", {
    "model": args.model, "type": "zero-shot LLM", "n_test": len(te),
    **{"test_" + k: v for k, v in classification_scores(te["label"], pred).items()}})
