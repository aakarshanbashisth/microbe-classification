# ════════════════════════════════════════════════════════════
#  augment.py  —  Run ONCE in terminal:  python augment.py
#
#  Takes every .png in your data/ subfolders and creates
#  7 new versions of each (3 rotations + 2 flips + dark + bright)
#
#  20 images per class  →  160 images per class automatically
# ════════════════════════════════════════════════════════════

from PIL import Image, ImageEnhance
import os

DATA_DIR = "data"   # the folder you created in Step 3

def augment_all():
    total = 0
    for folder_name in os.listdir(DATA_DIR):
        folder = os.path.join(DATA_DIR, folder_name)
        if not os.path.isdir(folder):
            continue

        # Only augment originals, not already-augmented files
        originals = [
            f for f in os.listdir(folder)
            if f.lower().endswith(".png") and "_aug_" not in f
        ]
        print(f"  {folder_name}: {len(originals)} originals found")

        for fname in originals:
            img  = Image.open(os.path.join(folder, fname))
            base = fname.replace(".png", "")

            versions = {
                "rot90":  img.rotate(90),
                "rot180": img.rotate(180),
                "rot270": img.rotate(270),
                "flipH":  img.transpose(Image.FLIP_LEFT_RIGHT),
                "flipV":  img.transpose(Image.FLIP_TOP_BOTTOM),
                "dark":   ImageEnhance.Brightness(img).enhance(0.8),
                "bright": ImageEnhance.Brightness(img).enhance(1.2),
            }

            for tag, new_img in versions.items():
                save_path = os.path.join(folder, f"{base}_aug_{tag}.png")
                new_img.save(save_path)
                total += 1

    print(f"\nDone!  {total} new augmented images created.")
    print("Now run:  python load_data.py")

if __name__ == "__main__":
    augment_all()