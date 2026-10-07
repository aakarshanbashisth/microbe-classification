import numpy as np
import joblib
import os
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing       import StandardScaler, LabelBinarizer
from sklearn.model_selection     import StratifiedKFold
from sklearn.metrics             import classification_report, confusion_matrix

MODEL_PATH = "model/plsda_model.pkl"


class PLSDAClassifier:
    """
    PLS-DA implemented as PLS Regression with a one-hot encoded Y matrix.
    This is the standard chemometrics approach, equivalent to MATLAB plsregress().
    """
    def __init__(self, n_components=3):
        self.pls      = PLSRegression(n_components=n_components, scale=False)
        self.scaler   = StandardScaler()
        self.lb       = LabelBinarizer()
        self.classes_ = None

    def fit(self, X, y):
        X_sc = self.scaler.fit_transform(X)
        Y    = self.lb.fit_transform(y)
        if Y.ndim == 1:                   # edge case: only 2 classes
            Y = np.column_stack([1 - Y, Y])
        self.pls.fit(X_sc, Y)
        self.classes_ = self.lb.classes_
        return self

    def _raw_scores(self, X):
        return self.pls.predict(self.scaler.transform(X))

    def predict(self, X):
        return self.classes_[np.argmax(self._raw_scores(X), axis=1)]

    # NEW — softmax gives more readable confidence spread
    def predict_proba(self, X):
        s = self._raw_scores(X)
        e = np.exp(s - s.max(axis=1, keepdims=True))  # numerically stable softmax
        return e / e.sum(axis=1, keepdims=True)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.array(y)))


def train(features: list, labels: list):
    """
    Train PLS-DA with 5-fold stratified cross-validation.
    Returns: (trained_model, cv_mean_accuracy, confusion_matrix)
    """
    X      = np.array(features)
    y      = np.array(labels)
    n_comp = min(3, len(set(labels)) - 1)   # PLS-DA rule: n_comp <= n_classes - 1

    model = PLSDAClassifier(n_components=n_comp)
    model.fit(X, y)

    # 5-fold cross-validation
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tr, te in cv.split(X, y):
        m = PLSDAClassifier(n_components=n_comp)
        m.fit(X[tr], y[tr])
        scores.append(m.score(X[te], y[te]))

    mean_acc = float(np.mean(scores))
    print(f"PLS-DA 5-fold CV: {mean_acc:.4f}  +/-  {np.std(scores):.4f}")
    y_pred = model.predict(X)
    print(classification_report(y, y_pred, target_names=model.classes_))

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print("PLS-DA model saved to model/plsda_model.pkl")
    return model, mean_acc, confusion_matrix(y, y_pred)


def predict(feature_vector):
    """
    Load saved PLS-DA and predict one image.
    Returns: (class_name, confidence_float, {class: probability_dict})
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("PLS-DA model not found — click Train PLS-DA first.")
    model = joblib.load(MODEL_PATH)
    X     = np.array(feature_vector).reshape(1, -1)
    cls   = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    return cls, float(proba.max()), dict(zip(model.classes_, proba))