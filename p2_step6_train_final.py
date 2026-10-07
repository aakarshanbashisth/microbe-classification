
import os, shutil
from PIL import Image, ImageEnhance
from feature_extractor import extract_features
from species_map       import SPECIES_MAP, get_all_species_by_type
import svm_classifier, plsda_classifier

DATA_DIR       = "C:\\AI-ML\\Data_Phase2"
AUG_DIR        = "C:\\AI-ML\\aug_temp_final"
MODEL_DIR      = "C:\\AI-ML\\model"
EVIDENCE_TYPES = ["microbe", "diatom", "pollen"]
IMG_EXTS       = ('.png', '.jpg', '.jpeg', '.bmp')

os.makedirs(MODEL_DIR, exist_ok=True)

def augment_all(ev_dir, species_list):
    """Augment ALL originals for one evidence type. Returns (files, labels)."""
    if os.path.exists(AUG_DIR):
        shutil.rmtree(AUG_DIR)
    os.makedirs(AUG_DIR)

    out_files, out_labels = [], []

    for sp in species_list:
        sp_dir = os.path.join(ev_dir, sp)
        if not os.path.isdir(sp_dir):
            continue
        dest_dir = os.path.join(AUG_DIR, sp)
        os.makedirs(dest_dir, exist_ok=True)

        for fname in sorted(os.listdir(sp_dir)):
            if not fname.lower().endswith(IMG_EXTS) or "_aug_" in fname:
                continue
            fpath = os.path.join(sp_dir, fname)
            base  = os.path.splitext(fname)[0]
            img   = Image.open(fpath)

            # Original
            orig_path = os.path.join(dest_dir, f"{base}.png")
            img.save(orig_path)
            out_files.append(orig_path); out_labels.append(sp)

            # 7 augmentations
            for tag, new_img in {
                "rot90" : img.rotate(90),
                "rot180": img.rotate(180),
                "rot270": img.rotate(270),
                "flipH" : img.transpose(Image.FLIP_LEFT_RIGHT),
                "flipV" : img.transpose(Image.FLIP_TOP_BOTTOM),
                "dark"  : ImageEnhance.Brightness(img).enhance(0.8),
                "bright": ImageEnhance.Brightness(img).enhance(1.2),
            }.items():
                p = os.path.join(dest_dir, f"{base}_aug_{tag}.png")
                new_img.save(p)
                out_files.append(p); out_labels.append(sp)

    return out_files, out_labels


print("=" * 55)
print("  PHASE 2 — STEP 6: Training Final Deployment Models")
print("  (trained on 100% of data — for demo only)")
print("=" * 55)

for ev_type in EVIDENCE_TYPES:
    ev_dir  = os.path.join(DATA_DIR, ev_type)
    species = get_all_species_by_type(ev_type)

    print(f"\n── {ev_type.upper()} ──────────────────────────────────")
    print("  Augmenting all data...")
    files, labels = augment_all(ev_dir, species)
    print(f"  Total augmented images: {len(files)}")

    print("  Extracting features...")
    features = [extract_features(f) for f in files]


    print("  Training SVM...")
    svm_classifier.train(features, labels)
    os.rename(
        os.path.join(MODEL_DIR, "svm_model.pkl"),
        os.path.join(MODEL_DIR, f"{ev_type}_svm_final.pkl"))
    os.rename(
        os.path.join(MODEL_DIR, "svm_labels.pkl"),
        os.path.join(MODEL_DIR, f"{ev_type}_svm_labels_final.pkl"))

    # Train PLS-DA
    print("  Training PLS-DA...")
    plsda_classifier.train(features, labels)
    os.rename(
        os.path.join(MODEL_DIR, "plsda_model.pkl"),
        os.path.join(MODEL_DIR, f"{ev_type}_plsda_final.pkl"))

    print(f"  Saved: {ev_type}_svm_final.pkl + {ev_type}_plsda_final.pkl")
    shutil.rmtree(AUG_DIR)

print(f"\n{'='*55}")
print("  All 6 final models saved to C:\\AI-ML\\model\\")
print("  model/")
for et in EVIDENCE_TYPES:
    print(f"    {et}_svm_final.pkl")
    print(f"    {et}_svm_labels_final.pkl")
    print(f"    {et}_plsda_final.pkl")
print("\n  Launch the demo:  python MainWindow_Phase2.py")
print(f"{'='*55}")
