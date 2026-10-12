"""Classical logistic-regression baseline."""
# The classical baseline uses logistic regression and records elapsed training time.
from time import perf_counter
from sklearn.linear_model import LogisticRegression

def train_classical(X_train, y_train):
    # CLASSICAL BASELINE: balanced Logistic Regression is the conventional comparison model.
    """Fit balanced logistic regression and return model plus wall-clock seconds."""
    # Balance class weights so the less common churn outcome still affects model fitting.
    model = LogisticRegression(class_weight="balanced", max_iter=1200, random_state=17)
    # Measure how long fitting takes so the dashboard can compare runtime later.
    start = perf_counter()
    # Learn the relationship between the prepared customer features and churn labels.
    model.fit(X_train, y_train)
    return model, perf_counter() - start
