"""Baseline type 2: fine-tune a contextual encoder. Saves the model for Integrated Gradients.
Usage: python src/baselines_encoder.py --model UBC-NLP/MARBERTv2"""
import argparse

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from common import ROOT, set_seed, load_split, label_names, save_result, classification_scores

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--epochs", type=int, default=3)
p.add_argument("--lr", type=float, default=2e-5)
p.add_argument("--bs", type=int, default=32)
p.add_argument("--max-len", type=int, default=128)
args = p.parse_args()
set_seed()
dev_ = "cuda" if torch.cuda.is_available() else "cpu"

labels = label_names()
l2i = {l: i for i, l in enumerate(labels)}
tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForSequenceClassification.from_pretrained(
    args.model, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=l2i).to(dev_)
tr, dv, te = load_split("train"), load_split("dev"), load_split("test")


def batches(df, shuffle):
    g = torch.Generator().manual_seed(42)
    idx = torch.randperm(len(df), generator=g).tolist() if shuffle else list(range(len(df)))
    for i in range(0, len(idx), args.bs):
        b = df.iloc[idx[i:i + args.bs]]
        enc = tok(list(b["text"]), truncation=True, max_length=args.max_len,
                  padding=True, return_tensors="pt")
        yield {k: v.to(dev_) for k, v in enc.items()}, torch.tensor([l2i[x] for x in b["label"]]).to(dev_)


@torch.no_grad()
def predict(df):
    model.eval()
    out = []
    for enc, _ in batches(df, False):
        out += model(**enc).logits.argmax(-1).tolist()
    return [labels[i] for i in out]


opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
steps = args.epochs * ((len(tr) + args.bs - 1) // args.bs)
sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
scaler = torch.amp.GradScaler(enabled=dev_ == "cuda")
best_f1 = -1
out_dir = ROOT / "models" / args.model.rstrip("/").split("/")[-1]
for ep in range(args.epochs):
    model.train()
    for enc, y in batches(tr, True):
        with torch.autocast(dev_, dtype=torch.float16, enabled=dev_ == "cuda"):
            loss = model(**enc, labels=y).loss
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
    f1 = classification_scores(dv["label"], predict(dv))["macro_f1"]
    print(f"epoch {ep + 1} dev macro-F1 {f1}")
    if f1 > best_f1:                       # keep the best epoch on dev
        best_f1 = f1
        model.save_pretrained(out_dir)
        tok.save_pretrained(out_dir)

model = AutoModelForSequenceClassification.from_pretrained(out_dir).to(dev_)
save_result(f"encoder_{out_dir.name}", {
    "model": args.model, "type": "fine-tuned encoder", "epochs": args.epochs, "lr": args.lr,
    "dev_macro_f1": best_f1,
    **{"test_" + k: v for k, v in classification_scores(te["label"], predict(te)).items()}})
