"""Measured metrics and timings for comparable held-out predictions."""
from time import perf_counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

def evaluate_model(model, X_test, y_test):
    """Return predictions, probabilities, metrics, confusion matrix and inference time."""
    start = perf_counter()
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= .5).astype(int)
    seconds = perf_counter() - start
    return {"predictions": predictions, "probabilities": probabilities,
      "accuracy": accuracy_score(y_test, predictions),
      "precision": precision_score(y_test, predictions, zero_division=0),
      "recall": recall_score(y_test, predictions, zero_division=0),
      "f1": f1_score(y_test, predictions, zero_division=0),
      "roc_auc": roc_auc_score(y_test, probabilities),
      "confusion": confusion_matrix(y_test, predictions), "roc": roc_curve(y_test, probabilities),
      "inference_seconds": seconds}
