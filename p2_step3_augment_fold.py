
import os, shutil
from PIL import Image, ImageEnhance

def augment_training_set(train_files, train_labels, aug_dir):
    """
    Augments training images ×8 into aug_dir/species/ folders.
    Returns (all_image_paths, all_labels) for the augmented training set.
    """
    if os.path.exists(aug_dir):
        shutil.rmtree(aug_dir)
    os.makedirs(aug_dir)

    out_files, out_labels = [], []

    for fpath, label in zip(train_files, train_labels):
        dest_class = os.path.join(aug_dir, label)
        os.makedirs(dest_class, exist_ok=True)

        img  = Image.open(fpath)
        base = os.path.splitext(os.path.basename(fpath))[0]

        # Save original
        orig_dest = os.path.join(dest_class, f"{base}.png")
        img.save(orig_dest)
        out_files.append(orig_dest)
        out_labels.append(label)

        # 7 augmented
        versions = {
            "rot90" : img.rotate(90),
            "rot180": img.rotate(180),
            "rot270": img.rotate(270),
            "flipH" : img.transpose(Image.FLIP_LEFT_RIGHT),
            "flipV" : img.transpose(Image.FLIP_TOP_BOTTOM),
            "dark"  : ImageEnhance.Brightness(img).enhance(0.8),
            "bright": ImageEnhance.Brightness(img).enhance(1.2),
        }
        for tag, new_img in versions.items():
            save_path = os.path.join(dest_class, f"{base}_aug_{tag}.png")
            new_img.save(save_path)
            out_files.append(save_path)
            out_labels.append(label)

    print(f"   Augmented: {len(out_files)} images from {len(train_files)} originals")
    return out_files, out_labels


if __name__ == "__main__":
    import json
    FOLDS_FILE = "C:\\AI-ML\\microbe_folds.json"
    AUG_DIR    = "C:\\AI-ML\\aug_temp_test"

    with open(FOLDS_FILE) as f:
        folds = json.load(f)

    fold = folds[0]
    files, labels = augment_training_set(
        fold["train_files"], fold["train_labels"], AUG_DIR)

    print(f"\nTest passed — {len(files)} training images in {AUG_DIR}")
    import shutil; shutil.rmtree(AUG_DIR)
    print("Temp folder cleaned up.  Proceed to Step 4.")
