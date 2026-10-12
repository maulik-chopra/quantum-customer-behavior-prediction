"""Measured metrics and timings for comparable held-out predictions."""
# Evaluation utilities measure held-out prediction quality and inference time.
from time import perf_counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

def evaluate_model(model, X_test, y_test):
    # EVALUATION: score customers the model did not train on, then calculate comparison metrics.
    """Return predictions, probabilities, metrics, confusion matrix and inference time."""
    # Time prediction separately from training.
    start = perf_counter()
    # Get the estimated probability of churn and apply a 0.5 cutoff for yes/no predictions.
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= .5).astype(int)
    seconds = perf_counter() - start
    # Return metrics, prediction details, confusion counts, ROC-curve points, and prediction time.
    return {"predictions": predictions, "probabilities": probabilities,
      "accuracy": accuracy_score(y_test, predictions),
      "precision": precision_score(y_test, predictions, zero_division=0),
      "recall": recall_score(y_test, predictions, zero_division=0),
      "f1": f1_score(y_test, predictions, zero_division=0),
      "roc_auc": roc_auc_score(y_test, probabilities),
      "confusion": confusion_matrix(y_test, predictions), "roc": roc_curve(y_test, probabilities),
      "inference_seconds": seconds}
