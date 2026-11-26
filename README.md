# Faithful explanations for code-switched sentiment (Topic 10)

GIU NLP course project, Milestone 1.

**Research question.** How faithful are LLM self-explanations and Integrated Gradients (IG)
attributions on code-switched sentiment data, and does rationale-supervised training improve
faithfulness?

> **Data:** EESA (Sherif and Sabty, github.com/ASTalaat/EESA-Corpus, 4,100 Egyptian Arabic-English
> code-switched comments, official train/dev/test files). The repo states no licence: coursework use, cite the source.
> **Rationales:** `data/rationales/rationales.csv` started as an AI-assisted first draft (Claude) and was then
> reviewed, audited and revised by the author (Mahmoud) together with classmates (35 of 50 rationales changed;
> 3 sentences were judged neutral by the reviewers, while EESA's gold labels are kept for all classification results).
> **Compute:** results below were produced on a 4-core CPU machine (no GPU), so the LLMs are small.

## What each part does (one line each)
| Step | File | Why |
|---|---|---|
| Data | `src/data.py` | load, clean, drop duplicates, split 80/10/10 (seed 42), print statistics |
| Embedding baseline | `src/baselines_embed.py` | frozen embedding model + logistic regression |
| Encoder baseline | `src/baselines_encoder.py` | fine-tune a BERT-style model; saved for IG |
| LLM baseline | `src/baselines_llm.py` | zero-shot LLM, label chosen from next-token probabilities |
| Rationales | `src/make_rationale_template.py` | 50 sentences for you to mark the words that justify the label |
| LLM faithfulness | `src/explain_llm.py` | LLM names its key words; we erase them and see if its confidence drops |
| Integrated Gradients | `src/ig_encoder.py` | gradient-based word importance on the fine-tuned encoder, same erasure test |

**Erasure scores (ERASER, DeYoung et al. 2020).** Comprehensiveness = p(full text) - p(text without the
explanation words): high means the words really mattered. Sufficiency = p(full text) - p(only the
explanation words): low means the words alone are enough. A random word set of the same size is the control.

## Commands (one per result; seeds fixed to 42)
```
pip install -r requirements.txt
git clone https://github.com/ASTalaat/EESA-Corpus data/raw/eesa-corpus
python src/data.py --source eesa --eesa-dir data/raw/eesa-corpus
python src/baselines_embed.py --model intfloat/multilingual-e5-base --prefix "query: "
python src/baselines_embed.py --model BAAI/bge-m3
python src/baselines_encoder.py --model UBC-NLP/MARBERTv2
python src/baselines_encoder.py --model aubmindlab/bert-base-arabertv02-twitter
python src/baselines_llm.py --model Qwen/Qwen2.5-1.5B-Instruct --n 200
python src/baselines_llm.py --model Qwen/Qwen3-1.7B --n 200   # on a GPU use Qwen2.5-7B-Instruct / Qwen3-8B
python src/make_rationale_template.py --n 50      # already done; reviewed rationales in data/rationales/rationales.csv
python src/explain_llm.py --model Qwen/Qwen3-1.7B --n 50
python src/ig_encoder.py --model-dir models/MARBERTv2 --n 50 --steps 32
python src/ig_encoder.py --model-dir models/bert-base-arabertv02-twitter --n 50 --steps 32
```
Or open `notebooks/run_all_colab.ipynb` on a free Colab T4. Results go to `results/*.json`.

**Report:** `report/report.tex` (ACL style files included; `cd report && pdflatex report.tex`, or upload the `report/` folder to Overleaf).

## Results (EESA test set, one run, seed 42; files in `results/`)
| Model | Type | Acc. | Macro-F1 |
|---|---|---|---|
| multilingual-e5-base | embedding + LR | .824 | .812 |
| BGE-M3 | embedding + LR | .830 | .817 |
| MARBERTv2 | fine-tuned encoder | .887 | .880 |
| AraBERTv02-twitter | fine-tuned encoder | .876 | .871 |
| Qwen2.5-1.5B-Instruct | zero-shot LLM (200 sent.) | .625 | .584 |
| Qwen3-1.7B | zero-shot LLM (200 sent.) | .710 | .672 |

| Explanation | Comprehensiveness (random) | Sufficiency (random) | F1 vs rationales |
|---|---|---|---|
| Qwen3-1.7B self-explanation | .282 (.363) | .336 (.359) | .410 |
| IG on MARBERTv2 | .178 (.059) | .163 (.293) | .351 |
| IG on AraBERT | .167 (.059) | .072 (.259) | .464 |

Reading: IG words matter to the encoder much more than random words; the small LLM's self-explanations do not beat random words.
Caveats: n=50, one run, different models are not directly comparable, rationales were AI-drafted, then reviewed, audited and revised by the author with classmates (no formal agreement score).
F1 uses the reviewed rationales (`src/rescore_rationale_f1.py`, no model re-run). IG top-k still uses the draft rationale lengths, so the IG erasure scores are from the draft; re-run `src/ig_encoder.py` to update them.

## Status and limits (honest)
- No GPU was available: LLMs are small (1.5-1.7B); 7-8B models are untested. Method v1 (rationale-supervised training) is not implemented.
- IG baseline = all word tokens replaced by `[PAD]`; completeness error is in the result files (MARBERT .03, AraBERT .10).
- LLM and IG faithfulness are measured on different models (each model's own probability).
- ArSAS stand-in code remains (`--source arsas`) but was not used for any reported number.

## AI use disclosure
Parts of the code and README were drafted with an AI assistant (Claude).
The report draft was also drafted with AI assistance; references were checked by opening their source pages.
The experiments were also run by the AI assistant, and the 50 rationales in `data/rationales/rationales.csv` were first drafted by it and then reviewed, audited and revised by the author with classmates (columns `source` and `reviewed`).
