
import os
from species_map import SPECIES_MAP

DATA_DIR = "C:\\AI-ML\\Data_Phase2"
EVIDENCE_TYPES = ["microbe", "diatom", "pollen"]

print("=" * 55)
print("  PHASE 2 — STEP 0: Folder Name Verification")
print("=" * 55)

all_ok = True

for ev_type in EVIDENCE_TYPES:
    ev_dir = os.path.join(DATA_DIR, ev_type)
    print(f"\n[{ev_type.upper()}]  {ev_dir}")

    if not os.path.isdir(ev_dir):
        print(f"  ✗  Folder not found — create it first")
        all_ok = False
        continue

    # Expected species for this evidence type
    expected = [sp for sp, (cls, et) in SPECIES_MAP.items() if et == ev_type]
    # Actual folders found
    found = [f for f in os.listdir(ev_dir)
             if os.path.isdir(os.path.join(ev_dir, f))]

    for sp in expected:
        sp_dir = os.path.join(ev_dir, sp)
        if os.path.isdir(sp_dir):
            imgs = [f for f in os.listdir(sp_dir)
                    if f.lower().endswith(('.png','.jpg','.jpeg','.bmp'))
                    and "_aug_" not in f]
            cls, _ = SPECIES_MAP[sp]
            print(f"  ✓  {sp:40s}  {len(imgs):3d} images  [{cls}]")
        else:
            print(f"  ✗  {sp:40s}  MISSING — create this folder")
            all_ok = False

    # Check for unexpected folders (typos etc.)
    unexpected = [f for f in found if f not in expected]
    for u in unexpected:
        print(f"  ⚠  '{u}' not in species_map.py — rename or add it")
        all_ok = False

print("\n" + "=" * 55)
if all_ok:
    print("  All folders verified.  Proceed to Step 1.")
else:
    print("  Fix the issues above before proceeding.")
print("=" * 55)
