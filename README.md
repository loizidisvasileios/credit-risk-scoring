# Credit Risk Scoring

Probability-of-default modelling on the *Give Me Some Credit* dataset. Four models are compared. Logistic Regression, Random Forest, XGBoost, LightGBM. The model with the best performance is served through an API with
per-applicant decline reasons, using SHAP values.



---

## Results

Test set, 30,000 applications:

| Model | PR-AUC | ROC-AUC | Gini | KS | Brier |
|---|---|---|---|---|---|
| LightGBM| 0.4082 | 0.8693 | 0.7386 | 0.5833 | 0.0780 |
| XGBoost | 0.4078 | 0.8688 | 0.7377 | 0.5807 | 0.0687 |
| Random Forest | 0.4046 | 0.8669 | 0.7338 | 0.5792 | 0.0489 |
| Logistic Regression | 0.3832 | 0.8602 | 0.7203 | 0.5621 | 0.1466 |

PR-AUC is the main metric

<p align="center">
  <img src="docs/assets/confusion_matrix.png" width="420" alt="Confusion matrix at the cost-based cut-off">
  <img src="docs/assets/target_balance.png" width="420" alt="Class balance in the training set">
</p>

---

## Cross-validation for model decision

The 150,000 applications are split 80/20, stratified on the target.


Cross-validation:

| | mean PR-AUC | standard error |
|---|---|---|
| LightGBM | 0.4034 | 0.0023 |
| XGBoost | 0.4017 | 0.0024 |
| Random Forest | 0.3988 | 0.0014 |
| Logistic Regression | 0.3715 | 0.0034 |

---

## Calibration

Isotonic calibration is fitted inside a 5-fold wrapper:

| | PR-AUC | KS | Brier | ECE |
|---|---|---|---|---|
| Uncalibrated | 0.4082 | 0.5833 | 0.0780 | 0.1294 |
| Isotonic | 0.4078 | 0.5818 | 0.0486 | 0.0026 |


<p align="center">
  <img src="docs/assets/calibration_curve.png" width="520" alt="Reliability curve before and after isotonic calibration">
</p>

The calibrated model decides;
`credit_model_uncalibrated.pkl` exists because TreeSHAP needs one booster while the
calibrated model is an ensemble of five. Isotonic regrerssion is based on a monotonic function, so we are able to do that.

---

## Values that are not true measurements

The delinquency counters stop at 17 and go to 96 and 98; revolving
utilisation reaches 50,708; `DebtRatio` holds a raw amount rather than a ratio
wherever income is missing. These are values that measure nothing, so they are handled separately from outliers:



---

## The decision cut-off

The threshold is decided by the difference between the cost of any decision.
Rejection of a good applicant: The cost is the margin `r`, times the Exposure At Default
Approval of a bad applicant: The cost is the Loss Given Default, times the Exposure At Default.
At a predicted probability `p`:

```
approve:  p * LGD * EAD          reject:  (1 - p) * r * EAD
```

`EAD` cancels, so one cut-off serves every loan size: reject above
`1 / (k + 1)`, with `k = LGD / r`. Here `k = 0.65 / 0.13 = 5` and the cut-off
is 0.1667 — 89.4% approved, 3.16% defaults among them.

---

## Explainability

SHAP contributions come from `TreeExplainer`, over a 5,000-application sample
of the test set.

<p align="center">
  <img src="docs/assets/shap_beeswarm.png" width="760" alt="SHAP summary plot">
</p>


---

## Running the pipeline

```bash
git clone https://github.com/<your-account>/credit-risk-scoring.git
cd credit-risk-scoring
python -m venv .venv && .venv/Scripts/activate      # Linux/macOS: source .venv/bin/activate
pip install -e ".[app,dev,data]"
python scripts/download_data.py
```

The stages run in order:

| Step | Command | Produces |
|---|---|---|
| 1 | `python scripts/run_eda.py` | Figures and tables |
| 2 | `python scripts/run_tuning.py` | Tuned parameters written back to the config |
| 3 | `python scripts/run_selection.py` | Per-fold scores and the pairwise differences behind the choice |
| 4 | `python scripts/run_evaluation.py` | Test metrics for the four models |
| 5 | `python scripts/run_calibration.py` | Calibration measurement and reliability curve |
| 6 | `python scripts/run_training.py` | The calibrated model, the uncalibrated and the metadata |
| 7 | `python scripts/run_explainability.py` | SHAP figures and reason codes |
| 8 | `python scripts/run_threshold.py` | Confusion matrix and cost-ratio sensitivity |


---

## App



<p align="center">
  <img src="docs/assets/dashboard.png" width="760" alt="The dashboard declining an application and naming the three largest contributions to that decision">
</p>

The API and the dashboard run together in one container:

```bash
docker compose up --build
```

`models/` is mounted read only rather than copied into the image, so the
pipeline has to have been run once before the container can score anything.

The dashboard is then at `http://127.0.0.1:7860`.

`POST /predict` takes an application and returns:

```json
{
  "probability": 0.1709,
  "score": 533,
  "decision": "reject",
  "threshold": 0.1667,
  "reason_codes": [
    {"feature": "NumberOfTime30-59DaysPastDueNotWorse", "value": 3.0, "contribution": 1.7465},
    {"feature": "DebtRatio", "value": 0.648, "contribution": 0.2260},
    {"feature": "MonthlyIncome", "value": 3325.0, "contribution": 0.1321}
  ]
}
```

Tests cover the schemas, the cut-off and the reason codes:

```bash
pytest
```

---

## Repository layout

```
config/config.yaml   Every constant the pipeline uses
src/credit_scoring/  Importable package: 13 modules, no entry points
scripts/             One command-line entry point per stage
app/                 service.py, the API and the dashboard
Dockerfile           Runs both of them as one service
tests/               API test
results/             Trial history
docs/assets/         Figures shown in this README
data/ models/ reports/   Generated locally.
```

Configuration is validated by pydantic on load.

---

## Data

*Give Me Some Credit*, Kaggle, 2011. 150,000 applications, 10 features, binary
target marking serious delinquency within two years.
## Licence

Released under the [MIT License](LICENSE).
