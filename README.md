# Images

Images come from **Recyclable and Household Waste Classification** on Kaggle ([dataset page](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification), by Alistair King).

# `split_dataset.py`

Splits each class’s `default` and `real_world` folders into train/test (no overlap between splits; both domains represented when present). Requires **Python 3** (stdlib only).

**Defaults:** `--source` is `images` (nested `images/images/<class>/...` is detected automatically); `--output` is `data`, creating `data/training data/` and `data/testing data/`; `--train-ratio` is `0.8`; `--seed` is `42`. Remove existing output dirs or use another `--output` if those folders already exist.

```bash
python3 split_dataset.py
python3 split_dataset.py --source images --output data --train-ratio 0.8 --seed 42 -v
python3 split_dataset.py --no-recursive   # images only directly under default/ and real_world/
```

| Option | Description |
|--------|-------------|
| `--source` | Root with one folder per class (default: `images`) |
| `--output` | Root for `training data` / `testing data` (default: `data`) |
| `--train-ratio` | Train fraction in `(0, 1)` |
| `--seed` | Random seed |
| `--no-recursive` | Do not recurse into subfolders of default/real_world |
| `-v` | Print per-class image counts |
