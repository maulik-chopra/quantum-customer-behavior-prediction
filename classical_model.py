"""Classical logistic-regression baseline."""
from time import perf_counter
from sklearn.linear_model import LogisticRegression

def train_classical(X_train, y_train):
    """Fit balanced logistic regression and return model plus wall-clock seconds."""
    model = LogisticRegression(class_weight="balanced", max_iter=1200, random_state=17)
    start = perf_counter()
    model.fit(X_train, y_train)
    return model, perf_counter() - start
