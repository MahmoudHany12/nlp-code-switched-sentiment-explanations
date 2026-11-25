# Topic 10: Faithful explanations for code-switched sentiment

GIU NLP course project, Milestone 1.

## Setup
```
pip install -r requirements.txt
```

## Data
1. Put the EESA file in `data/raw/` (for example `data/raw/eesa.csv`).
2. Run:
```
python src/prepare_data.py --input data/raw/eesa.csv
```
This cleans the text, removes duplicates, splits 80/10/10 (stratified, seed 42)
and writes `data/processed/{train,dev,test}.csv` and `stats.json`.

If the columns are not called `text` and `label`, add
`--text-col <name> --label-col <name>`.

- Licence of EESA: TODO (ask the instructor / check the corpus files).

## Reproducibility
- Fixed seed: 42 (`--seed` to change).
- Exact package versions are in `requirements.txt`.

## AI use disclosure
Parts of the code and README were drafted with an AI assistants .
