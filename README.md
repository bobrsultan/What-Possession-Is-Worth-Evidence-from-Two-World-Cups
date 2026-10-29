# What Possession Is Worth: Evidence from Two World Cups

Code and data for the paper "What Possession Is Worth: Evidence from Two World Cups", Journal of High School Science.

## Contents

- `data/`: match, team and player statistics, pre-tournament Elo ratings, and the result tables the scripts produce
- `figures/`: figures used in the paper
- `test_underdog.py`: builds the underdog match dataset (Elo gap of 50 or more) and fits the possession x directness interaction model
- `run_ablation.py`: possession ablation (R-squared, partial F-test, 10-fold cross-validated RMSE, with and without possession)
- `run_dml_possession.py`: double machine learning estimates of the possession effect (ridge and random forest, 3 seeds)

## Running

Run from the repository root, in this order:

```
pip install -r requirements.txt
python test_underdog.py
python run_ablation.py
python run_dml_possession.py
```

Each script reads from and writes to `data/`.
