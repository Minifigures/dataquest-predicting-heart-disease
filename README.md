# Predicting Heart Disease

Binary classification on the UCI Heart Failure Prediction dataset (918 patients), tuned for recall rather than accuracy, with a data quality finding that mattered more than the model choice.

Dataquest guided project, Data Scientist in Python path.

## Results

Tuned KNN: **k=17, Manhattan distance (p=1), distance-weighted**, selected from 64 configurations by 5-fold cross-validated recall. Tuning lifted CV recall from 0.8818 (untuned k=5) to 0.9163.

Held-out test set, 184 patients:

| Metric | Value |
|---|---|
| Accuracy | 0.9076 |
| Precision | 0.9126 |
| **Recall** | **0.9216** |
| F1 | 0.9171 |
| ROC AUC | 0.9449 |
| Majority-class baseline accuracy | 0.5543 |

Confusion matrix: 73 TN, 9 FP, **8 FN**, 94 TP.

| Model | Accuracy | Precision | Recall | ROC AUC | Missed cases |
|---|---|---|---|---|---|
| KNN (tuned) | 0.9076 | 0.9126 | 0.9216 | 0.9449 | 8 |
| Logistic regression | 0.8913 | 0.8868 | 0.9216 | 0.9327 | 8 |
| Random forest | 0.8804 | 0.8846 | 0.9020 | 0.9371 | 10 |

Logistic regression ties KNN exactly on recall, missing the same eight patients, for 1.6 points of accuracy. On a tie in the metric that matters clinically, the interpretable model is arguably the better deployment choice: it gives a coefficient per predictor a clinician can challenge, where KNN can only report that seventeen similar patients were mostly sick.

## The data quality finding

`isnull()` reported zero missing values. In fact **172 patients (18.7%) had cholesterol recorded as 0**, and one had a resting blood pressure of 0. Neither is physiologically possible; both are absent measurements written as zeros, which means they were about to be fed into a distance metric where a patient with no cholesterol reading looks like the healthiest patient in the dataset.

The missingness is strongly informative:

| | Heart disease rate | n |
|---|---|---|
| Cholesterol present | 47.7% | 746 |
| Cholesterol recorded as 0 | **88.4%** | 172 |
| Overall | 55.3% | 918 |

Both reflex responses would have been wrong. Dropping those rows discards 18.7% of the data along with a strong signal; imputing them silently buries that signal in the median. The notebook replaces the zeros with `NaN` so they are imputed honestly inside the pipeline **and** keeps a `CholesterolMissing` flag so the model can use the fact of the absence. The random forest ranks that flag 9th of 17 features (importance 0.0409).

Why the signal exists is a question the data cannot answer. The plausible reading is a recording artefact at one contributing hospital rather than anything clinical, which is a caution against treating it as a finding about patients.

## What carries the signal

| Feature | Importance |
|---|---|
| ST_Slope_Up | 0.1640 |
| ST_Slope_Flat | 0.1199 |
| Oldpeak | 0.1139 |
| MaxHR | 0.1031 |
| ExerciseAngina_Y | 0.0850 |
| Age | 0.0766 |

The top five all come from the exercise stress test. The resting measurements rank below them, which is consistent with the clinical literature and a useful sanity check that the model is keying on something real.

## Threshold as a policy choice

| Threshold | Recall | Accuracy | Missed cases | False alarms |
|---|---|---|---|---|
| 0.60 | 0.9020 | 0.9130 | 10 | 6 |
| 0.50 (default) | 0.9216 | 0.9076 | 8 | 9 |
| 0.35 | 0.9510 | 0.8859 | 5 | 16 |
| 0.20 | 0.9706 | 0.8098 | 3 | 32 |
| 0.10 | 0.9804 | 0.7880 | 2 | 37 |

A false negative is a patient with heart disease sent home; a false positive is further testing that was not needed. Those costs are not symmetric, so 0.50 is an arbitrary default rather than a clinical decision. The table exists to hand that judgement to whoever should be making it.

## Method

1. **Profile and quality check**, which is where the disguised missing values turned up.
2. **EDA** of every predictor against the target, categorical rates and numeric distributions split by outcome.
3. **Handle impossible values**: zeros to `NaN`, plus an explicit missingness flag.
4. **One-hot encode** before the split (safe: row-wise and deterministic), **defer imputation and scaling** into the pipeline (unsafe: they compute cross-row statistics).
5. **Stratified 80/20 split**, preserving the 55.3% positive rate.
6. **Leak-free `Pipeline`** of median imputer, standard scaler and estimator, refitted on each CV fold so fold statistics never see held-out rows.
7. **`GridSearchCV` over 64 configurations** scored on recall, so `refit` selects the model that misses the fewest sick patients.
8. **Compare** against logistic regression and a random forest through the identical pipeline on the identical split.
9. **Threshold sweep**, reported with its own caveat.

## Limitations

- The threshold table is computed on the test set. Reading a threshold off it and then reporting the same test metrics would be selection leakage; in practice the threshold belongs on a validation split.
- 184 test patients is small. One extra false negative moves recall about a point, so the KNN-versus-logistic-regression gap on recall (zero) is well inside the noise. Repeated or nested CV would give an honest variance estimate.
- Median imputation across 18.7% of cholesterol values compresses that feature's variance. The flag mitigates this; it does not repair the measurement.
- No calibration check. KNN probabilities from 17 neighbours are coarse by construction.
- This is the pooled UCI heart dataset from several hospitals in the late 1980s. It does not represent current populations or diagnostic practice, so none of these numbers estimate real-world screening performance.
- Nothing here is a diagnostic tool. It is a classification exercise on a public teaching dataset.

## Stack

Python, pandas, NumPy, scikit-learn (Pipeline, SimpleImputer, StandardScaler, KNeighborsClassifier, LogisticRegression, RandomForestClassifier, GridSearchCV, StratifiedKFold), matplotlib, seaborn.

## Files

- `notebook.ipynb`: the full analysis with executed outputs and figures.