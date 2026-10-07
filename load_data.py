# ════════════════════════════════════════════════════════════
#  load_data.py  —  Run ONCE in terminal:  python load_data.py
#
#  Reads ALL .png images from your data/ subfolders and
#  inserts them into MySQL with their class label.
#  Subfolder name  =  class label  (automatically)
# ════════════════════════════════════════════════════════════

from DataBaseClass import Database
import os

DATA_DIR = "data"

# ⚠️  Change YOUR_MYSQL_PASSWORD to your actual MySQL root password
db = Database("localhost", "root", "Forsci@0420", "new_schema")
db.connect()
db.create_table()

total = 0
for folder_name in os.listdir(DATA_DIR):
    folder = os.path.join(DATA_DIR, folder_name)
    if not os.path.isdir(folder):
        continue

    label  = folder_name                                # folder name IS the label
    images = [f for f in os.listdir(folder)
              if f.lower().endswith(".png")]

    print(f"Loading {len(images):3d} images for class '{label}' ...")

    for fname in images:
        with open(os.path.join(folder, fname), "rb") as f:
            db.insert_photo(f.read(), label)
        total += 1

db.close()
print(f"\nAll done!  {total} images loaded into MySQL.")
print("Open the app:  python Hackathon.py")