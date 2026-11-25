"""Baseline type 1: frozen text-embedding model + logistic regression.
Usage: python src/baselines_embed.py --model intfloat/multilingual-e5-base"""
import argparse

from sklearn.linear_model import LogisticRegression

from common import SEED, set_seed, load_split, save_result, classification_scores

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--prefix", default="", help="e5 models want 'query: '")
args = p.parse_args()
set_seed()

from sentence_transformers import SentenceTransformer
enc = SentenceTransformer(args.model)
tr, dv, te = load_split("train"), load_split("dev"), load_split("test")
emb = lambda df: enc.encode([args.prefix + t for t in df["text"]], batch_size=64,
                            normalize_embeddings=True, show_progress_bar=True)
Xtr, Xdv, Xte = emb(tr), emb(dv), emb(te)

best = None
for C in [0.1, 1, 10]:                      # pick C on the dev set
    clf = LogisticRegression(C=C, max_iter=2000, random_state=SEED).fit(Xtr, tr["label"])
    f1 = classification_scores(dv["label"], clf.predict(Xdv))["macro_f1"]
    if best is None or f1 > best[0]:
        best = (f1, C, clf)
f1, C, clf = best
name = args.model.split("/")[-1]
save_result(f"embed_{name}", {"model": args.model, "type": "embedding+LR", "C": C,
                              "dev_macro_f1": f1,
                              **{"test_" + k: v for k, v in
                                 classification_scores(te["label"], clf.predict(Xte)).items()}})
