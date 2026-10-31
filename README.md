# Credit Risk Scoring

Probability-of-default modelling on the *Give Me Some Credit* dataset, from
exploratory analysis through hyperparameter search to a served scoring API with
per-applicant explanations.

Four models are compared under an identical cross-validation protocol, the
winner is selected before the test set is opened, the accept/reject cut-off is
derived from the cost of the two error types rather than chosen by hand, and
every declined application comes with the reasons behind it.

---

## Results

Held-out test set, 30,000 applications never used in any decision:

| Model | PR-AUC | ROC-AUC | Gini | KS | Brier |
|---|---|---|---|---|---|
| **XGBoost** | **0.4105** | **0.8699** | **0.7398** | 0.5818 | **0.0485** |
| LightGBM | 0.4098 | 0.8696 | 0.7392 | **0.5833** | 0.0486 |
| Random Forest | 0.4051 | 0.8673 | 0.7346 | 0.5793 | 0.0489 |
| Logistic Regression | 0.3473 | 0.8364 | 0.6729 | 0.5163 | 0.1656 |

PR-AUC is the headline metric: only 6.68% of applications default, so a measure
that ignores the easy negatives is the one that reflects the task.

At the derived cut-off of 0.1667 the policy approves **89.9%** of applications
and carries a **3.26%** default rate among those approved.

<p align="center">
  <img src="docs/assets/confusion_matrix.png" width="420" alt="Confusion matrix at the cost-based cut-off">
  <img src="docs/assets/target_balance.png" width="420" alt="Class balance in the training set">
</p>

---

## How the model was chosen

The 150,000 applications are split 80/20, stratified on the target. **The test
set takes no part in any decision** — not in the exploratory analysis, not in
the outlier bounds or imputation medians, not in tuning, and not in model
selection.

Inside the training set, 5-fold stratified cross-validation answers two
questions:

```
150,000 applications
   |
   +-- 120,000 TRAINING -------------------------+     30,000 TEST
   |                                             |     (sealed)
   |  [1] which hyperparameters                  |
   |      Optuna, 30 trials x 4 models x 5 folds |
   |      = 600 fits, all discarded              |
   |                                             |
   |  [2] which model                            |
   |      4 tuned configurations, same folds     |
   |      = 20 fits, all discarded               |
   |      -> XGBoost                             |
   |                                             |
   |  [3] one fit on all 120,000 -> the model    |
   +---------------------------------------------+
                                                 |
                                   one measurement +--> reported above
```

Cross-validation here is a measuring instrument, not a training step: every
model fitted inside a fold is thrown away.

**XGBoost and LightGBM tied.** Their cross-validated PR-AUC differed by 0.0004,
less than one standard error, and they swapped places depending on the metric.
The tie was broken on reproducibility, not performance. The choice was locked in
before the test set was read, and the test result — which happens to agree — did
not influence it.

### Calibration was measured, then dropped

The decision rule below needs probabilities that are right in level, not merely
in rank, so isotonic calibration was fitted and compared against the raw model.

| | PR-AUC | Brier | Mean predicted probability |
|---|---|---|---|
| Uncalibrated | 0.4105 | 0.0485 | 0.0676 |
| Isotonic | 0.4111 | 0.0485 | 0.0676 |

The tuned model already predicted an average default probability of 0.0676
against an observed rate of 0.0668, a 1.2% discrepancy, because the search
settled on `scale_pos_weight ≈ 1.0` and never distorted the probability scale.
Calibration left the Brier score unchanged and was therefore dropped, with the
experiment kept in the repository as the evidence.

<p align="center">
  <img src="docs/assets/calibration_curve.png" width="520" alt="Reliability curve before and after isotonic calibration">
</p>

---

## The decision cut-off

Rejecting a sound applicant forgoes the margin `r`; approving one who defaults
loses `LGD` of the exposure. Comparing the expected cost of each decision for an
applicant with predicted probability `p`:

```
approve:  p * LGD * EAD          reject:  (1 - p) * r * EAD
```

Rejecting is cheaper once `p > 1 / (k + 1)`, where `k = LGD / r`. The exposure
cancels, so the cut-off does not depend on loan size. With `LGD = 0.65` and
`r = 0.13`, `k = 5` and the cut-off is **0.1667**.

`k` is an assumption rather than a measurement — the dataset carries no loan
amounts or recoveries — so the policy is published across `k = 2` to `k = 10` in
`reports/tables/threshold_sensitivity.csv`. Approval rate falls from 95.2% to
81.9% across that range.

---

## Explainability

SHAP contributions come from `TreeExplainer`, exact rather than sampled, over a
5,000-application sample of the test set.

<p align="center">
  <img src="docs/assets/shap_beeswarm.png" width="760" alt="SHAP summary plot">
</p>

Revolving utilisation separates applicants on a continuous scale, while the
three delinquency counters act as near-switches: their absence gives everyone a
small, uniform benefit, their presence penalises heavily and unevenly.

The same values drive the decline reasons returned by the API, so the
explanation a credit analyst reads is the one the model actually used.

---

## Running the pipeline

```bash
git clone https://github.com/<your-account>/credit-risk-scoring.git
cd credit-risk-scoring
python -m venv .venv && .venv/Scripts/activate      # Linux/macOS: source .venv/bin/activate
pip install -e ".[app,dev,data]"
python scripts/download_data.py
```

The stages run in order; each writes its results to `reports/` and prints them.

| Step | Command | Produces | Time |
|---|---|---|---|
| 1 | `python scripts/run_eda.py` | 25 figures, 7 tables | 1 min |
| 2 | `python scripts/run_tuning.py` | Trial history, tuned parameters written back to the config | 85 min |
| 3 | `python scripts/run_comparison.py` | Cross-validated comparison of the four models | 12 min |
| 4 | `python scripts/run_evaluation.py` | Test metrics for the four models | 5 min |
| 5 | `python scripts/run_calibration.py` | Calibration experiment and reliability curve | 7 min |
| 6 | `python scripts/run_training.py` | `models/credit_model.pkl` and its metadata | 2 min |
| 7 | `python scripts/run_explainability.py` | SHAP figures and reason codes | 2 min |
| 8 | `python scripts/run_threshold.py` | Confusion matrix and sensitivity analysis | 1 min |

Steps 7 and 8 load the saved model, so step 6 must have run.

Hyperparameters are not hard-coded: step 2 writes them into `config/config.yaml`
with a dated provenance comment, and the full trial history is kept under
`results/tuning/` so every published number can be traced to the run that
produced it.

---

## Serving

The scoring logic lives in `app/service.py`. The API exposes it over HTTP and
the dashboard is a pure client — it holds no model and makes no decisions,
exactly as a production front end would not.

```bash
uvicorn app.api:app --port 8000     # first terminal
python app/dashboard.py            # second terminal
```

| Address | |
|---|---|
| `http://127.0.0.1:8050` | Dashboard |
| `http://127.0.0.1:8000/docs` | Interactive API documentation, generated from the schemas |

`POST /predict` takes an application and returns:

```json
{
  "probability": 0.1664,
  "score": 534,
  "decision": "approve",
  "threshold": 0.1667,
  "reason_codes": []
}
```

The score is the points scale lenders publish: it rises as risk falls, and every
20 points doubles the odds of repayment. Reason codes are returned only for a
decline, since an approval has none to give. A scored application takes
70–150 ms, the longer path being the one that computes SHAP values.

```bash
pytest
```

---

## Repository layout

```
config/config.yaml   Every constant the pipeline uses, validated on load
src/credit_scoring/  Importable package: 12 modules, no entry points
scripts/             One command-line entry point per stage
app/                 service.py, the API over it, and the dashboard over that
tests/               API test suite
results/tuning/      Full Optuna trial history, tracked in version control
docs/assets/         Figures shown in this README
data/ models/ reports/   Generated locally, not tracked
```

Configuration is validated by pydantic on load, so a malformed value fails
immediately with the field named, rather than surfacing as a strange result
forty minutes into a tuning run.

---

## Data

*Give Me Some Credit*, Kaggle, 2011. 150,000 applications, 10 features, binary
target marking serious delinquency within two years. The competition rules must
be accepted on Kaggle before the download script can fetch it.

## Licence

Released under the [MIT License](LICENSE).
