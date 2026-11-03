# Convertible Bond Return Predictability with Machine Learning

## What this project demonstrates

- Cross-sectional ranking of predictors within each week
- 13 model families: OLS, Lasso, Ridge, Elastic Net, PCR, PLS, gradient boosting, random forest, and neural networks with one to five hidden layers
- Validation-based hyperparameter choice
- Out-of-sample R-squared against a zero-return forecast
- Weekly prediction and a simple equal-weighted top-minus-bottom portfolio summary

This is a compact demonstrating framework. It does not implement every detail in my work, including every predictor, robustness check, or economic restriction.

## Repository layout

```text
.
├── README.md
├── requirements.txt
├── data/
│   └── README.md
└── src/
    ├── __init__.py
    └── study.py
```

## Setup

Python 3.10 or later is recommended.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Input data

Prepare a CSV with one row per bond-week and these columns:

| Column | Meaning |
|---|---|
| `date` | Week-ending date |
| `bond_id` | Convertible bond identifier |
| `ret_fwd` | Next week's bond return, in decimal form |
| predictor columns | Information available as of `date` (already lagged/aligned) |

Do not include identifiers, dates, or future information among the predictors. See [`data/README.md`](data/README.md) for the data contract and leakage checks.

## Quick start

```python
import pandas as pd
from src.study import run_study

panel = pd.read_csv("data/panel.csv", parse_dates=["date"])
result = run_study(panel)
print(result["metrics"])
result["predictions"].to_csv("predictions.csv", index=False)
```

By default, the study uses 2017–2018 for training, 2019 for validation, and 2020–2022 for testing. To try it on another dataset, pass explicit date boundaries to `run_study`.

## Notes on interpretation

- `ret_fwd` should be the return from the next period, paired with predictors known at the current week.
- Predictor ranking is performed within each date, which avoids using later observations to scale earlier data.
- The validation period is used only for model selection. The selected model is refit on the training and validation observations before testing.
- The simple portfolio output is an illustration, not a transaction-cost-aware investment strategy. It omits short-sale constraints, turnover, liquidity, survivorship, and execution effects.

## References

- Gu, S., Kelly, B., and Xiu, D. (2020). “Empirical Asset Pricing via Machine Learning.” *The Review of Financial Studies*, 33(5), 2223–2273. https://doi.org/10.1093/rfs/hhaa009

## License

The code in this repository is an independent educational implementation. The paper and its text are copyrighted by their respective rights holders. Do not add the paper PDF or proprietary CSMAR data to a public repository. Confirm data permissions before publishing any dataset or results derived from it.
