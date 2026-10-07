
import json, os, shutil
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score,
                             cohen_kappa_score, classification_report)
from feature_extractor    import extract_features
from p2_step3_augment_fold import augment_training_set
from species_map           import SPECIES_MAP
import svm_classifier, plsda_classifier

FOLDS_DIR  = "C:\\AI-ML"
AUG_DIR    = "C:\\AI-ML\\aug_temp"
RESULTS    = "C:\\AI-ML\\p2_fold_results.json"
EVIDENCE_TYPES = ["microbe", "diatom", "pollen"]

all_results = {}

for ev_type in EVIDENCE_TYPES:
    folds_path = os.path.join(FOLDS_DIR, f"{ev_type}_folds.json")
    if not os.path.exists(folds_path):
        print(f"✗ {folds_path} not found — run Step 2 first"); continue

    with open(folds_path) as f:
        folds = json.load(f)

    svm_results, plsda_results = [], []

    print(f"\n{'='*55}")
    print(f"  EVIDENCE TYPE: {ev_type.upper()}")
    print(f"{'='*55}")

    for fold in folds:
        fold_num = fold["fold"]
        print(f"\n  Fold {fold_num} / 5")

        # Augment training set
        print("  Augmenting training images...")
        train_files, train_labels = augment_training_set(
            fold["train_files"], fold["train_labels"], AUG_DIR)

        # Extract features
        print("  Extracting training features...")
        train_features = [extract_features(f) for f in train_files]

        print("  Extracting test features...")
        test_features = [extract_features(f) for f in fold["test_files"]]
        test_labels   = fold["test_labels"]

        # Train SVM
        print("  Training SVM...")
        svm_classifier.train(train_features, train_labels)

        # Train PLS-DA
        print("  Training PLS-DA...")
        plsda_classifier.train(train_features, train_labels)

        # Evaluate both models
        for model_name in ["SVM", "PLS-DA"]:
            if model_name == "SVM":
                preds_raw = [svm_classifier.predict(f)   for f in test_features]
            else:
                preds_raw = [plsda_classifier.predict(f) for f in test_features]

            preds = [p[0] for p in preds_raw]

            # Auto-enrich: add class label from species_map
            enriched = []
            for sp_pred, true_sp in zip(preds, test_labels):
                cls_pred, _ = SPECIES_MAP.get(sp_pred,  ("unknown", "unknown"))
                cls_true, _ = SPECIES_MAP.get(true_sp,  ("unknown", "unknown"))
                enriched.append({
                    "true_species" : true_sp,
                    "true_class"   : cls_true,
                    "pred_species" : sp_pred,
                    "pred_class"   : cls_pred,
                    "species_correct" : sp_pred == true_sp,
                    "class_correct"   : cls_pred == cls_true,
                })

            acc   = accuracy_score(test_labels, preds)
            f1    = f1_score(test_labels, preds, average="weighted",
                             zero_division=0)
            kappa = cohen_kappa_score(test_labels, preds)

            result = {
                "fold"          : fold_num,
                "model"         : model_name,
                "evidence_type" : ev_type,
                "accuracy"      : round(acc,   4),
                "f1"            : round(f1,    4),
                "kappa"         : round(kappa, 4),
                "predictions"   : enriched,
            }

            if model_name == "SVM":
                svm_results.append(result)
            else:
                plsda_results.append(result)

            print(f"    {model_name}  acc={acc:.2%}  F1={f1:.4f}  κ={kappa:.4f}")

        shutil.rmtree(AUG_DIR)

    all_results[ev_type] = {"svm": svm_results, "plsda": plsda_results}

    # Print summary for this evidence type
    print(f"\n  ── {ev_type.upper()} SUMMARY ──")
    for model_name, results in [("SVM", svm_results), ("PLS-DA", plsda_results)]:
        accs   = [r["accuracy"] for r in results]
        f1s    = [r["f1"]       for r in results]
        kappas = [r["kappa"]    for r in results]
        print(f"  {model_name:8s}  acc={np.mean(accs):.2%}±{np.std(accs):.2%}"
              f"  F1={np.mean(f1s):.4f}  κ={np.mean(kappas):.4f}")

# Save all results
with open(RESULTS, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n{'='*55}")
print(f"  All results saved to {RESULTS}")
print("  Proceed to Step 5.")
print(f"{'='*55}")
