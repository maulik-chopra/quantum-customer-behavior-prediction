# Code explanation guide

## File responsibilities

- `app.py`: Streamlit layout, dataset selection, interactive analysis, model training controls, prediction form, CSV export and experiment report.
- `src/data_processing.py`: target detection, synthetic demo generator, label validation, train/test split and preprocessing pipelines.
- `src/classical_model.py`: class-balanced Logistic Regression training and elapsed-time measurement.
- `src/quantum_model.py`: training-only PCA, angle scaling, four-qubit circuit and Qiskit variational classifier on Aer simulation.
- `src/evaluation.py`: held-out probability predictions, thresholded labels, standard classification metrics and inference time.
- `requirements.txt`: bounded major-version dependency ranges.
- `VIVA_PREPARATION.md`: speakable presentation outline and defense questions.

## Data flow

CSV or synthetic demo rows → target selection → target-label validation → remove identifier and target from feature matrix → stratified raw train/test split → fit imputers, one-hot encoder and scaler on train only → transform held-out customers → fit Logistic Regression and optionally the PCA/VQC pipeline → calculate metrics against the same held-out labels → display charts, single-customer and batch predictions, and downloadable report.

The transformer and quantum PCA are never fitted using held-out labels. PCA is fitted only on the training feature matrix. The analysis visualization's PCA is descriptive and separate from the model pipeline.

## Main functions

### `detect_target(frame)`

Input is a pandas DataFrame. It checks case-insensitive common target names and returns a matching original column name or `None`.

### `clean_target(series)`

Maps recognized churn/retention spellings into integers. It stops with a clear error for unknown or missing labels because silently guessing labels could invert the task.

### `make_demo_data(rows, seed)`

Creates reproducible, artificial Telco-like data. A logistic rule creates plausible label patterns. It exists only to demonstrate the app when the real dataset is unavailable.

### `prepare_data(frame, target_column, test_size, seed)`

Validates a binary target, removes ID-like columns, identifies numeric/categorical features, splits rows with stratification, and returns transformed arrays plus the fitted transformer, feature names, and raw partitions. `fit_transform` is applied to training rows only; test rows call `transform` only. That separation prevents data leakage.

### `train_classical(X_train, y_train)`

Fits Logistic Regression with `class_weight="balanced"`, which weights the less common class more strongly. It returns the fitted estimator and measured wall-clock fit time.

### `train_quantum(X_train, y_train, X_test, max_rows)`

Fits PCA on training rows and retains four components. A min-max scaler maps them to input angles. `ZZFeatureMap` encodes values using rotations and pairwise ZZ interactions; linear entanglement couples neighboring qubits. `RealAmplitudes` adds trainable rotation layers and entangling gates. VQC adjusts those trainable angles with COBYLA to reduce classification loss. Aer executes a quantum circuit simulation on a classical computer; this is not a quantum hardware run. The fit uses a bounded sample for runtime.

### `evaluate_model(model, X_test, y_test)`

Measures probability prediction time, uses 0.5 as the classification threshold, and computes accuracy, precision, recall, F1 and ROC-AUC. It also returns confusion matrix and ROC curve coordinates. `y_test` is never used for fitting.

## Metrics in plain language

- **Accuracy:** fraction of all predictions that are correct. It can hide poor detection of a minority class.
- **Precision:** of customers flagged as churners, the fraction who actually churned.
- **Recall:** of customers who churned, the fraction successfully flagged. A false negative is a churner the model missed.
- **F1:** harmonic mean of precision and recall; balances the two.
- **ROC-AUC:** how often a random churner is ranked above a random retained customer across thresholds. 0.5 is chance ranking; it is not a chosen-threshold accuracy.
- **Risk thresholds:** the dashboard's high/medium cutoffs translate probabilities into categories. They are adjustable business rules, not model certainty.

## Important limitations

The synthetic records do not establish real customer behavior. One random split gives uncertain estimates. Correlated, duplicated, or shifted data can lower future performance. A small PCA representation can discard information. Variational quantum training can be slow and results depend on optimizer settings and simulator behavior. A simulator is classical computation that models quantum circuits. This project does not demonstrate speedup or quantum advantage. Churn associations are predictive/statistical and not causal explanations.

## Common errors

Read `README.md` for setup and dataset steps. In particular, inspect the selected binary target, check CSV headers for spaces/renamed fields, and use the on-screen error for Qiskit dependency/API failures. The classical model should still run when Qiskit is unavailable.
