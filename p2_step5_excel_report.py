
import json
import numpy as np
import pandas as pd

RESULTS = "C:\\AI-ML\\p2_fold_results.json"
EXCEL   = "C:\\AI-ML\\FIU_Phase2_Results.xlsx"

with open(RESULTS) as f:
    data = json.load(f)

writer = pd.ExcelWriter(EXCEL, engine="openpyxl")

# ── Sheet 1: Overall fold summary ───────────────────────────────
summary_rows = []
for ev_type, models in data.items():
    for model_name, results in [("SVM", models["svm"]),
                                 ("PLS-DA", models["plsda"])]:
        for r in results:
            summary_rows.append({
                "Evidence Type" : ev_type,
                "Model"         : model_name,
                "Fold"          : r["fold"],
                "Accuracy"      : f"{r['accuracy']:.2%}",
                "F1 Score"      : round(r["f1"],    4),
                "Kappa"         : round(r["kappa"], 4),
            })
        # Mean row
        accs   = [r["accuracy"] for r in results]
        f1s    = [r["f1"]       for r in results]
        kappas = [r["kappa"]    for r in results]
        summary_rows.append({
            "Evidence Type" : ev_type.upper(),
            "Model"         : f"{model_name} MEAN ± STD",
            "Fold"          : "—",
            "Accuracy"      : f"{np.mean(accs):.2%} ± {np.std(accs):.2%}",
            "F1 Score"      : f"{np.mean(f1s):.4f} ± {np.std(f1s):.4f}",
            "Kappa"         : f"{np.mean(kappas):.4f} ± {np.std(kappas):.4f}",
        })

pd.DataFrame(summary_rows).to_excel(
    writer, sheet_name="Fold Summary", index=False)

# ── Sheet 2: Per-image predictions ──────────────────────────────
img_rows = []
for ev_type, models in data.items():
    svm_results   = models["svm"]
    plsda_results = models["plsda"]

    for fold_idx, (svm_r, pls_r) in enumerate(
            zip(svm_results, plsda_results)):

        for svm_pred, pls_pred in zip(
                svm_r["predictions"], pls_r["predictions"]):

            img_rows.append({
                "Evidence Type"     : ev_type,
                "Fold"              : svm_r["fold"],
                "True Species"      : svm_pred["true_species"],
                "True Class"        : svm_pred["true_class"],
                "SVM Species"       : svm_pred["pred_species"],
                "SVM Class"         : svm_pred["pred_class"],
                "SVM Species ✓"     : "✓" if svm_pred["species_correct"] else "✗",
                "SVM Class ✓"       : "✓" if svm_pred["class_correct"]   else "✗",
                "PLSDA Species"     : pls_pred["pred_species"],
                "PLSDA Class"       : pls_pred["pred_class"],
                "PLSDA Species ✓"   : "✓" if pls_pred["species_correct"] else "✗",
                "PLSDA Class ✓"     : "✓" if pls_pred["class_correct"]   else "✗",
                "Both Species Agree": "✓" if svm_pred["pred_species"] ==
                                             pls_pred["pred_species"] else "✗",
            })

pd.DataFrame(img_rows).to_excel(
    writer, sheet_name="Per Image Results", index=False)

# ── Sheet 3: Class-level accuracy ───────────────────────────────
# How well does species prediction translate to correct class?
class_rows = []
for ev_type, models in data.items():
    for model_name, results in [("SVM", models["svm"]),
                                 ("PLS-DA", models["plsda"])]:
        all_preds = [p for r in results for p in r["predictions"]]
        sp_correct  = sum(1 for p in all_preds if p["species_correct"])
        cls_correct = sum(1 for p in all_preds if p["class_correct"])
        total       = len(all_preds)

        class_rows.append({
            "Evidence Type"   : ev_type,
            "Model"           : model_name,
            "Total Tested"    : total,
            "Species Correct" : sp_correct,
            "Species Accuracy": f"{sp_correct/total:.2%}",
            "Class Correct"   : cls_correct,
            "Class Accuracy"  : f"{cls_correct/total:.2%}",
            "Class Gain"      : f"+{(cls_correct-sp_correct)/total:.2%}",
        })

pd.DataFrame(class_rows).to_excel(
    writer, sheet_name="Class Accuracy", index=False)

writer.close()
print(f"Excel report saved to {EXCEL}")
print("Proceed to Step 6 (final model training for demo).")
