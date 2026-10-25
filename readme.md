
# Stellar Classifier

**Engineering-first multiclass classification for the [Kaggle Playground Series S6E6: Predicting Stellar Class](https://www.kaggle.com/competitions/playground-series-s6e6).**

This project builds a reproducible machine-learning workflow to classify astronomical observations as **GALAXY**, **QSO** (quasar), or **STAR**. It extends a histogram-based gradient boosting baseline with data-contract checks, domain-informed feature engineering, fold-safe preprocessing, cross-validation, interpretability, saved experiments, and reusable inference.

> **Project status:** The notebook contains an executed full-data, five-fold experiment and a validated Kaggle submission file. Reported results below are **out-of-fold validation results**, not a Kaggle public/private leaderboard score.

## Results

| Metric | Five-fold out-of-fold result |
|---|---:|
| **Balanced accuracy (competition metric)** | **0.95131 (95.13%)** |
| Overall accuracy | 0.9641 (96.41%) |
| Multiclass log loss | 0.10766 |
| Recall — GALAXY | 0.9752 |
| Recall — QSO | 0.9622 |
| Recall — STAR | 0.9166 |

**Validation:** Stratified five-fold cross-validation, random seed `42`, on all **577,347** labeled training records. The test set contains **247,435** unlabeled records.

The notebook also explores class-specific decision multipliers. On its OOF policy-audit partition, balanced accuracy changed from **0.95095** (raw) to **0.95444** (adjusted). This is an exploratory internal check, **not** an independent external test or a replacement for the primary OOF score.

## Project Structure

```text
stellar-classifier/
├── data/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   └── astro_classifier.ipynb
├── models/
├── outputs/
├── README.md
├── requirements.txt
└── .gitignore
```

The notebook is stored in `notebooks/`, while the original datasets are maintained in `data/`. Trained models and generated evaluation artifacts are stored separately for organization and reproducibility.

## Dataset

The dataset comes from the Kaggle competition linked above.

### Input Features

| Category | Features | Description |
|---|---|---|
| Sky Position | `alpha`, `delta` | Astronomical coordinates |
| Photometry | `u`, `g`, `r`, `i`, `z` | Magnitude measurements across photometric bands |
| Redshift | `redshift` | Observed spectral shift |
| Categorical | `spectral_type`, `galaxy_population` | Astronomical descriptors |
| Identifier | `id` | Record identifier, excluded from training |
| Target | `class` | GALAXY, QSO, or STAR |

### Class Distribution

| Class | Training Rows | Share |
|---|---:|---:|
| GALAXY | 377,480 | 65.38% |
| QSO | 117,143 | 20.29% |
| STAR | 82,724 | 14.33% |

Because the target classes are unevenly represented, **balanced accuracy** is used as the primary evaluation metric.

## Methodology

### 1. Data Validation

Before model development, input datasets are validated against an expected schema.

Validation checks include:

- Expected columns and data types.
- Missing and infinite numerical values.
- Duplicate or null identifiers.
- Valid target classifications.
- Training and test feature consistency.
- Overlapping identifiers between datasets.

These checks help identify data quality issues before they affect downstream modeling.

### 2. Exploratory Data Analysis

Exploratory analysis examines:

- Target class distributions.
- Redshift distributions.
- Photometric color indices.
- Relationships between measurements and stellar classes.
- Potential differences between training and test feature distributions.

Reproducible sampling is used for visualizations to reduce computational overhead while preserving the full dataset for model training.

### 3. Feature Engineering

The pipeline expands the original astronomical measurements with domain-informed features.

**Photometric Color Indices**

- `u_g`
- `g_r`
- `r_i`
- `i_z`
- `g_z`
- `u_r`

**Magnitude Statistics**

- `mag_mean`
- `mag_std`
- `mag_range`
- `color_spread`

These features represent relationships between photometric measurements and variations across bands.

Feature transformations are implemented as reusable functions and applied consistently during training and inference.

The complete model uses **20 features**, including original measurements, engineered numerical variables, and categorical descriptors.

### 4. Model Development

The classification pipeline uses:

**HistGradientBoostingClassifier**

Preprocessing is implemented through scikit-learn's `ColumnTransformer` and `Pipeline`.

Numerical features pass directly to the estimator, while categorical descriptors are transformed using `OrdinalEncoder`.

Categorical encoding is fitted independently within each training fold to prevent validation data from influencing preprocessing.

### Model Configuration

| Parameter | Value |
|---|---|
| Estimator | HistGradientBoostingClassifier |
| Cross-Validation | Stratified 5-Fold |
| Random Seed | 42 |
| Maximum Iterations | 200 |
| Learning Rate | 0.1 |
| Maximum Leaf Nodes | 31 |
| L2 Regularization | 0.0 |
| Early Stopping | Enabled |

### 5. Cross-Validation

Stratified five-fold cross-validation is used to evaluate generalization performance.

Each fold:

1. Fits an independent preprocessing and classification pipeline.
2. Generates probabilities for held-out validation observations.
3. Calculates balanced accuracy, accuracy, log loss, and class recall.
4. Generates test-set probabilities.
5. Saves the trained pipeline.

Out-of-fold predictions are collected for every training observation.

Test probabilities are averaged across fold models to produce ensemble predictions.

### 6. Model Evaluation

The notebook evaluates classification performance using:

- Balanced accuracy.
- Overall accuracy.
- Multiclass log loss.
- Precision, recall, and F1-score.
- Confusion matrices.
- Per-class recall.
- Prediction confidence and error analysis.

Both standard and normalized confusion matrices are generated to examine misclassification patterns.

### 7. Decision Policy Optimization

The notebook evaluates class-specific probability multipliers as an alternative to standard maximum-probability classification.

Candidate weights are selected using a portion of OOF predictions and evaluated on a separate policy-audit partition.

This experiment explores whether alternative classification decisions improve balanced accuracy without retraining the underlying estimator.

The policy-audit results are treated as exploratory and do not replace the primary cross-validation score.

### 8. Model Interpretability

Permutation importance is used to investigate how individual features contribute to classification performance.

Feature values are shuffled within held-out validation observations, and the resulting change in balanced accuracy is recorded.

This helps identify variables associated with predictive performance while accounting for limitations related to correlated photometric measurements.

### 9. Reproducibility and Experiment Tracking

Experiment artifacts are saved separately from the source notebook.

The workflow records:

- Model hyperparameters.
- Random seed.
- Feature definitions.
- Class ordering.
- Cross-validation metrics.
- Dependency versions.
- Decision-policy settings.
- Serialized preprocessing and model pipelines.

The experiment manifest is stored as JSON for reproducibility and comparison across experiments.

### 10. Reusable Inference

The project includes a reusable inference function that:

1. Loads saved experiment metadata.
2. Validates incoming observations.
3. Applies consistent feature engineering.
4. Loads serialized fold pipelines.
5. Averages predicted probabilities.
6. Applies the saved decision policy.
7. Returns predicted stellar classes.

A reproducibility check confirms that predictions from saved models match previously generated predictions on a sample of test observations.

## Running the Project

### 1. Download the Data

Download `train.csv` and `test.csv` from the [Kaggle competition](https://www.kaggle.com/competitions/playground-series-s6e6/data).

Place both files inside the `data/` directory.

### 2. Install Dependencies

```bash
python -m pip install numpy pandas scikit-learn matplotlib joblib ipython notebook
```

The executed notebook recorded Python 3.13.5, pandas 2.2.3, and scikit-learn 1.6.1.

### 3. Launch Jupyter Notebook

From the repository root:

```bash
jupyter notebook notebooks/astro_classifier.ipynb
```

### 4. Configure the Experiment

```python
SEED = 42
SMOKE_TEST = False
SAVE_MODELS = True
RUN_IMPORTANCE = True
```

Set `SMOKE_TEST = True` to execute a smaller development experiment before running the full dataset.

### 5. Run the Notebook

Execute all cells in order.

The notebook creates the necessary model and output directories and saves evaluation artifacts automatically.

## Generated Artifacts

| File | Description |
|---|---|
| `outputs/fold_metrics.csv` | Per-fold validation metrics |
| `outputs/oof_predictions.csv` | Out-of-fold predictions and probabilities |
| `outputs/experiment_manifest.json` | Experiment configuration and metadata |
| `outputs/submission.csv` | Validated Kaggle submission |
| `models/model_fold_*.joblib` | Serialized preprocessing and classification pipelines |

## Future Improvements

### Model Benchmarking

Evaluate alternative gradient boosting implementations, including LightGBM, XGBoost, and CatBoost, using consistent cross-validation splits.

### Feature Ablation

Compare the baseline feature set against expanded engineered features to measure their individual contribution to balanced accuracy.

### Hyperparameter Optimization

Explore learning rates, tree complexity, regularization, and boosting iterations through controlled experiments.

### Automated Testing

Introduce unit and integration tests for data validation, feature transformations, model loading, and inference.

### MLOps Integration

Extend the notebook into a more complete machine learning engineering project through:

- Dependency pinning.
- Automated testing with GitHub Actions.
- Experiment tracking and model versioning.
- Standalone batch inference.
- REST API deployment.
- Input data and prediction monitoring.

## Limitations

This project uses a synthetically generated Kaggle competition dataset. Strong cross-validation results do not necessarily represent real-world astronomical deployment performance.

Class-specific decision adjustments require additional independent validation before being treated as confirmed performance improvements.

Permutation importance provides model diagnostics rather than causal explanations, particularly when engineered features are correlated.

## Conclusion

This project demonstrates an engineering-focused approach to multiclass classification, emphasizing reproducibility, input validation, domain-informed feature engineering, leakage-safe preprocessing, model evaluation, and reusable inference.

The resulting workflow provides a foundation for further model experimentation and production-oriented machine learning development.

## Competition Reference

[Kaggle Playground Series S6E6 — Predicting Stellar Class](https://www.kaggle.com/competitions/playground-series-s6e6)
