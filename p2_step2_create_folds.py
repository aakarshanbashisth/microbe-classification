
import os, json
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from species_map import SPECIES_MAP, get_all_species_by_type

DATA_DIR       = "C:\\AI-ML\\Data_Phase2"
FOLDS_DIR      = "C:\\AI-ML"
EVIDENCE_TYPES = ["microbe", "diatom", "pollen"]

print("=" * 55)
print("  PHASE 2 — STEP 2: Creating Stratified 5-Fold Splits")
print("=" * 55)

for ev_type in EVIDENCE_TYPES:
    ev_dir   = os.path.join(DATA_DIR, ev_type)
    species  = get_all_species_by_type(ev_type)

    all_files, all_labels = [], []

    for sp in sorted(species):
        sp_dir = os.path.join(ev_dir, sp)
        if not os.path.isdir(sp_dir):
            continue
        for fname in sorted(os.listdir(sp_dir)):
            if fname.lower().endswith(('.png','.jpg','.jpeg','.bmp')) \
               and "_aug_" not in fname:
                all_files.append(os.path.join(sp_dir, fname))
                all_labels.append(sp)   # label = species name

    print(f"\n── {ev_type.upper()} ──  {len(all_files)} originals  "
          f"({len(set(all_labels))} species)")

    skf   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = []

    for fold_num, (train_idx, test_idx) in enumerate(
            skf.split(all_files, all_labels), start=1):

        train_files  = [all_files[i] for i in train_idx]
        train_labels = [all_labels[i] for i in train_idx]
        test_files   = [all_files[i] for i in test_idx]
        test_labels  = [all_labels[i] for i in test_idx]

        folds.append({
            "fold"         : fold_num,
            "train_files"  : train_files,
            "train_labels" : train_labels,
            "test_files"   : test_files,
            "test_labels"  : test_labels
        })

        test_counts = Counter(test_labels)
        print(f"  Fold {fold_num}: train={len(train_files)}  "
              f"test={len(test_files)}  "
              f"species in test={len(test_counts)}")

    folds_path = os.path.join(FOLDS_DIR, f"{ev_type}_folds.json")
    with open(folds_path, "w") as f:
        json.dump(folds, f, indent=2)
    print(f"  Saved → {folds_path}")

print(f"\n{'='*55}")
print("  All 3 fold files saved.  Proceed to Step 3.")
print(f"{'='*55}")
