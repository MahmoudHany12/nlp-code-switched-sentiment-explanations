# Faithful explanations for code-switched sentiment (Topic 10)

GIU NLP course project, Milestone 1.

**Research question.** How faithful are LLM self-explanations and Integrated Gradients (IG)
attributions on code-switched sentiment data, and does rationale-supervised training improve
faithfulness?

> **STAND-IN DATA.** The EESA corpus has not been received yet. All numbers produced with
> `--source arsas` use the public ArSAS Arabic tweet corpus (`arbml/ArSAS` on Hugging Face) as a
> stand-in. Check its licence on the dataset card before reporting. Re-run with EESA
> (see "Switching to EESA") before final results. Human rationales must be written by the student.

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
python src/data.py --source arsas
python src/baselines_embed.py --model intfloat/multilingual-e5-base --prefix "query: "
python src/baselines_embed.py --model BAAI/bge-m3
python src/baselines_encoder.py --model UBC-NLP/MARBERTv2
python src/baselines_encoder.py --model aubmindlab/bert-base-arabertv02-twitter
python src/baselines_llm.py --model Qwen/Qwen2.5-7B-Instruct --n 300
python src/baselines_llm.py --model Qwen/Qwen3-8B --n 300
python src/make_rationale_template.py --n 50      # fill it, save as data/rationales/rationales.csv
python src/explain_llm.py --model Qwen/Qwen2.5-7B-Instruct --n 50
python src/ig_encoder.py --model-dir models/MARBERTv2 --n 50 --steps 32
```
Or open `notebooks/run_all_colab.ipynb` on a free Colab T4. Results go to `results/*.json`.

## Switching to EESA
```
python src/data.py --source csv --input data/raw/eesa.csv --text-col <text column> --label-col <label column>
```
Labels must be English words whose first tokens differ (e.g. negative / neutral / positive), because the
LLM prompt uses them. Rename them in the CSV otherwise. Then re-run the commands above.

## Status and limits (honest)
- The pipeline was tested end to end only on tiny random models and fake data (logic check), not yet on
  the real models, so no real results exist until the commands above are run on Colab.
- IG baseline = all word tokens replaced by `[PAD]`; completeness error is reported per run.
- LLM and IG faithfulness are measured on different models (the LLM's and the encoder's own probabilities).
- Method v1 (rationale-supervised training) is not implemented yet; the limitation experiment is the
  faithfulness comparison above. See the report draft in `report/`.

## AI use disclosure
Parts of the code and README were drafted with an AI assistants .
The report draft was also drafted with AI assistance; references were checked by opening their source pages.
