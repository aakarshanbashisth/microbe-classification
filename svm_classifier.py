import numpy as np
import joblib
import os
from sklearn.svm              import SVC
from sklearn.preprocessing    import StandardScaler, LabelEncoder
from sklearn.model_selection  import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline         import Pipeline
from sklearn.metrics          import classification_report, confusion_matrix, accuracy_score

MODEL_PATH = "model/svm_model.pkl"
LABEL_PATH = "model/svm_labels.pkl"


def train(features: list, labels: list):
    """
    Train SVM on the feature list + label list from the database.
    Automatically finds the best hyperparameters via grid search.
    Saves the trained model to disk.

    Returns: (trained_model, test_accuracy, confusion_matrix)
    """
    X  = np.array(features)
    le = LabelEncoder()
    y  = le.fit_transform(labels)

    # 80% for training, 20% for testing
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Pipeline: scale features first, then train SVM
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(kernel="rbf", probability=True, class_weight="balanced"))
    ])

    # Try different values of C and gamma to find the best combination
    param_grid = {
        "svm__C":     [0.1, 1, 10, 100],
        "svm__gamma": ["scale", "auto", 0.001, 0.01]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(pipeline, param_grid, cv=cv,
                      scoring="accuracy", n_jobs=-1, verbose=0)
    gs.fit(X_tr, y_tr)

    best   = gs.best_estimator_
    y_pred = best.predict(X_te)
    acc    = accuracy_score(y_te, y_pred)
    cm     = confusion_matrix(y_te, y_pred)

    print(f"Best params  : {gs.best_params_}")
    print(f"Test accuracy: {acc:.4f}")
    print(classification_report(y_te, y_pred, target_names=le.classes_))

    os.makedirs("model", exist_ok=True)
    joblib.dump(best, MODEL_PATH)
    joblib.dump(le,   LABEL_PATH)
    print("SVM model saved to model/svm_model.pkl")
    return best, acc, cm


def predict(feature_vector):
    """
    Load saved SVM and predict the class of one image's features.
    Returns: (class_name, confidence_float, {class: probability_dict})
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("SVM model not found — click Train SVM first.")
    model = joblib.load(MODEL_PATH)
    le    = joblib.load(LABEL_PATH)
    X     = np.array(feature_vector).reshape(1, -1)
    pred  = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    cls   = le.inverse_transform([pred])[0]
    return cls, float(proba.max()), dict(zip(le.classes_, proba))