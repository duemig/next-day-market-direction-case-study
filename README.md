# Next-day equity market direction from 27 indicators

Case study completed in February 2020 for a technical interview at Union Investment. The firm supplied an anonymised dataset and asked for a model of next-day market direction, an explanation of which indicators matter, and an assessment of how portfolio managers could use it. The suggested effort was two to three hours.

Code: [linear_code.ipynb](linear_code.ipynb) (logistic regression) and [non_linear_code.ipynb](non_linear_code.ipynb) (random forest). Slides for the interview: [Presentation_linear.pdf](Presentation_linear.pdf), [Presentation_non_linear.pdf](Presentation_non_linear.pdf).

## Data

| Item | Detail |
|---|---|
| Market | A global equity market, not named by the provider |
| Period | 4 January 1960 to 27 January 2020, daily |
| Observations | 15,119 (15,117 after dropping 2 incomplete rows) |
| Inputs | 27 anonymised indicators (`Indikator1` to `Indikator27`) |
| Target | 1 if the market rises on the following day, 0 otherwise; already aligned by the provider |
| Class balance | 53% up days overall, 54.4% in the test period |

The dataset belongs to the firm and is not included, so the notebooks cannot be re-run from this repository. Their saved outputs are the results below.

## Method

- **Split:** chronological, first 90% for training, last 1,512 days (23 January 2014 onwards) for testing, no shuffling.
- **Preprocessing variants:** raw; winsorised at three interquartile ranges beyond the quartiles; standardised; winsorised and standardised. All transforms are fitted on the training set only.
- **Class imbalance:** SMOTE oversampling of down days, applied to the training set only.
- **Logistic regression:** fitted on each variant, with and without SMOTE.
- **Feature selection:** recursive feature elimination inside a pipeline, with the number of indicators (1, 2, 5 or 10) chosen by walk-forward cross-validation (`TimeSeriesSplit`, 8 folds).
- **Random forest:** 500 trees, entropy criterion, square-root feature sampling, tuned by the same walk-forward scheme; feature selection by importance threshold.
- **Interpretation:** permutation importance.

## Results

Test-set accuracy, 1,512 days. Always predicting "up" scores 54.4%.

| Model | Accuracy |
|---|---|
| Logistic regression, raw inputs | 67.1% |
| Logistic regression, winsorised | 67.0% |
| Logistic regression, standardised | 70.4% |
| Logistic regression, winsorised and standardised | 69.8% |
| Logistic regression with SMOTE (four variants) | 67.4% to 70.2% |
| Logistic regression with RFE (2 indicators selected) | 70.6% |
| Random forest | 70.0% |
| Random forest with SMOTE | 70.2% |
| Random forest with importance-based selection (2 indicators) | 69.8% |

Both model families select the same two indicators (11 and 22), and these carry almost all of the predictive power. A logistic regression on those two inputs does as well as any larger model.

## Assessment

A 70% hit rate for the next-day direction of a broad equity market is much higher than is usually found in practice. The most likely explanations lie in the data rather than the model: an indicator that uses information from the target day, or asynchronous closing times across the markets that make up a global index. The indicators are anonymised, so this could not be checked. The result should be read as a property of the supplied dataset, not as evidence of a tradable signal.

Other limitations:

- One chronological split; no confidence intervals or tests of the difference to the base rate.
- Accuracy only; no strategy returns, transaction costs or turnover.
- The test period starts in 2014, so the estimate covers a single, mostly rising market regime.

## Reproducing

The notebooks were run in early 2020 with Python 3.6 and the package versions of that time (not recorded). Some calls have since changed:

- `imblearn`: `fit_sample` is now `fit_resample`.
- `GridSearchCV(iid=...)` was removed in scikit-learn 0.24.
- `feature_engine.outlier_removers` moved to `feature_engine.outliers`.

`requirements.txt` lists the packages used.

## Contents

| File | Description |
|---|---|
| `linear_code.ipynb` | Data checks, preprocessing variants, logistic regression, SMOTE, RFE, permutation importance |
| `non_linear_code.ipynb` | Random forest with walk-forward tuning, SMOTE and feature selection |
| `Presentation_linear.pdf`, `Presentation_non_linear.pdf` | Interview slides for a non-technical audience |
| `work-flow.png`, `walk-forward.png`, `scaling.png` | Figures used in the notebooks |
| `requirements.txt` | Python packages used |
