
import os
from species_map import SPECIES_MAP, get_all_species_by_type

DATA_DIR       = "C:\\AI-ML\\Data_Phase2"
EVIDENCE_TYPES = ["microbe", "diatom", "pollen"]
MIN_IMAGES     = 10   # minimum originals per species for reliable CV

print("=" * 55)
print("  PHASE 2 — STEP 1: Original Image Verification")
print("=" * 55)

grand_total = 0

for ev_type in EVIDENCE_TYPES:
    ev_dir   = os.path.join(DATA_DIR, ev_type)
    species  = get_all_species_by_type(ev_type)
    ev_total = 0

    print(f"\n── {ev_type.upper()} ──────────────────────────────────")

    for sp in sorted(species):
        sp_dir = os.path.join(ev_dir, sp)
        if not os.path.isdir(sp_dir):
            print(f"  {sp}: FOLDER MISSING")
            continue

        all_files = os.listdir(sp_dir)
        originals = [f for f in all_files
                     if f.lower().endswith(('.png','.jpg','.jpeg','.bmp'))
                     and "_aug_" not in f]
        aug_files = [f for f in all_files if "_aug_" in f]
        cls, _    = SPECIES_MAP[sp]

        flag = ""
        if len(aug_files) > 0:
            flag = f"  ⚠ {len(aug_files)} aug files found — delete them"
        if len(originals) < MIN_IMAGES:
            flag += f"  ✗ below minimum ({MIN_IMAGES})"

        print(f"  {sp:40s}  {len(originals):3d} orig  aug:{len(aug_files):3d}"
              f"  [{cls}]{flag}")
        ev_total += len(originals)

    after_aug = ev_total * 8
    n_folds   = 5
    test_per_fold  = ev_total // n_folds
    train_per_fold = ev_total - test_per_fold

    print(f"\n  Subtotal originals : {ev_total}")
    print(f"  After ×8 aug train : ~{train_per_fold * 8}")
    print(f"  Test per fold      : ~{test_per_fold} (never augmented)")
    grand_total += ev_total

print(f"\n{'='*55}")
print(f"  Grand total originals : {grand_total}")
print(f"  If aug files = 0 everywhere, proceed to Step 2.")
print(f"{'='*55}")
