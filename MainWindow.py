import tkinter as tk
from tkinter import filedialog, messagebox
from DataBaseClass import Database
from PIL import Image, ImageTk
import io

class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Forensic Microbial Classifier  —  NFSU Delhi")
        self.root.geometry("700x610")
        self.root.configure(bg="#1a1a2e")

        # ── Database connection ─────────────────────────────────────────────
        # ⚠️  CHANGE "YOUR_MYSQL_PASSWORD" to your actual MySQL root password
        self.db = Database("localhost", "root", "Forsci@0420", "new_schema")
        self.db.connect()
        self.db.create_table()

        # ── Title labels ────────────────────────────────────────────────────
        tk.Label(root,
                 text="Forensic Microbial Classification System",
                 font=("Arial", 14, "bold"), bg="#1a1a2e", fg="white"
                 ).pack(pady=(14, 2))
        tk.Label(root,
                 text="SVM + PLS-DA  |  NFSU Delhi Campus  |  FIU Presentation 2025",
                 font=("Arial", 9), bg="#1a1a2e", fg="#8888aa"
                 ).pack(pady=(0, 12))

        # ── Buttons ─────────────────────────────────────────────────────────
        btn = dict(width=40, font=("Arial", 10), pady=5, bd=0, cursor="hand2")

        tk.Button(root, text="  Insert Labelled Image(s)",
                  command=self.insert_photo,
                  bg="#3d3d5c", fg="white", **btn).pack(pady=3)

        tk.Button(root, text="  Train SVM Model  (Python)",
                  command=self.train_svm,
                  bg="#5b2d8e", fg="white", **btn).pack(pady=3)

        tk.Button(root, text="  Train PLS-DA Model  (Python / MATLAB)",
                  command=self.train_plsda,
                  bg="#0e6b6b", fg="white", **btn).pack(pady=3)

        tk.Button(root, text="  Classify Unknown Image",
                  command=self.classify_image,
                  bg="#1a6b2e", fg="white", **btn).pack(pady=3)

        tk.Button(root, text="  View Database Summary",
                  command=self.view_summary,
                  bg="#3d3d5c", fg="white", **btn).pack(pady=3)

        # ── Image preview area ──────────────────────────────────────────────
        self.image_label = tk.Label(root, bg="#1a1a2e")
        self.image_label.pack(pady=6)

        # ── Results panel ────────────────────────────────────────────────────
        self.result_var = tk.StringVar(value="Ready.  Insert images to begin.")
        tk.Label(root,
                 textvariable=self.result_var,
                 font=("Courier", 10), bg="#0d0d1a", fg="#00ff88",
                 wraplength=650, justify="left",
                 pady=10, padx=14
                 ).pack(pady=4, fill="x", padx=22)

    # ── INSERT PHOTOS ──────────────────────────────────────────────────────
    def insert_photo(self):
        paths = filedialog.askopenfilenames(
            title="Select microscopy images",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not paths:
            return

        # Pop-up to choose the class label
        win = tk.Toplevel(self.root)
        win.title("Select class label")
        win.geometry("330x300")
        win.configure(bg="#1a1a2e")
        win.grab_set()                # block the main window until done

        tk.Label(win, text="Select the morphological class:",
                 font=("Arial", 11, "bold"),
                 bg="#1a1a2e", fg="white").pack(pady=12)

        classes   = ["gram_pos_cocci", "gram_neg_bacilli",
                     "gram_pos_bacilli", "gram_neg_cocci", "fungi"]
        class_var = tk.StringVar(value=classes[0])
        for c in classes:
            tk.Radiobutton(win, text=c, variable=class_var, value=c,
                           font=("Courier", 10), bg="#1a1a2e", fg="#aaffcc",
                           selectcolor="#2a2a4e",
                           activebackground="#1a1a2e").pack(anchor="w", padx=30)

        def confirm():
            chosen = class_var.get()
            n = 0
            for fp in paths:
                with open(fp, "rb") as f:
                    self.db.insert_photo(f.read(), chosen)
                n += 1
            self.result_var.set(f"Inserted {n} image(s) labelled as '{chosen}'")
            win.destroy()

        tk.Button(win, text="Confirm and Insert", command=confirm,
                  bg="#1a6b2e", fg="white",
                  font=("Arial", 10), width=22).pack(pady=12)

    # ── TRAIN SVM ─────────────────────────────────────────────────────────
    def train_svm(self):
        self.result_var.set("Training SVM model — please wait...")
        self.root.update()
        rows = self.db.fetch_all_labelled()
        if len(rows) < 6:
            messagebox.showerror("Not enough data",
                "Need at least 6 labelled images. Insert more images first.")
            return
        try:
            from feature_extractor import extract_features
            import svm_classifier
            features = [extract_features(blob) for blob, _ in rows]
            labels   = [lbl for _, lbl in rows]
            _, acc, _ = svm_classifier.train(features, labels)
            self.result_var.set(
                f"SVM training complete\n"
                f"Test accuracy  :  {acc:.2%}\n"
                f"Model saved    :  model/svm_model.pkl")
        except Exception as e:
            messagebox.showerror("Training Error", str(e))

    # ── TRAIN PLS-DA ──────────────────────────────────────────────────────
    def train_plsda(self):
        self.result_var.set("Training PLS-DA model — please wait...")
        self.root.update()
        rows = self.db.fetch_all_labelled()
        if len(rows) < 6:
            messagebox.showerror("Not enough data",
                "Need at least 6 labelled images. Insert more images first.")
            return
        try:
            from feature_extractor import extract_features
            import plsda_classifier
            features = [extract_features(blob) for blob, _ in rows]
            labels   = [lbl for _, lbl in rows]
            _, acc, _ = plsda_classifier.train(features, labels)
            self.result_var.set(
                f"PLS-DA training complete\n"
                f"CV accuracy    :  {acc:.2%}\n"
                f"Model saved    :  model/plsda_model.pkl")
        except Exception as e:
            messagebox.showerror("Training Error", str(e))

    # ── CLASSIFY UNKNOWN IMAGE ────────────────────────────────────────────
    def classify_image(self):
        path = filedialog.askopenfilename(
            title="Select unknown microscopy image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return

        try:
            from feature_extractor import extract_features
            feat = extract_features(path)
        except Exception as e:
            messagebox.showerror("Error", f"Feature extraction failed:\n{e}")
            return

        lines = ["Classification Results", "-" * 38]

        try:
            import svm_classifier
            cls, conf, probs = svm_classifier.predict(feat)
            lines.append(f"SVM     =>  {cls}  ({conf:.1%})")
            for lbl, p in sorted(probs.items(), key=lambda x: -x[1])[:3]:
                lines.append(f"            {lbl}: {p:.1%}")
        except FileNotFoundError:
            lines.append("SVM     =>  model not trained yet")

        lines.append("")

        try:
            import plsda_classifier
            cls2, conf2, probs2 = plsda_classifier.predict(feat)
            lines.append(f"PLS-DA  =>  {cls2}  ({conf2:.1%})")
            for lbl, p in sorted(probs2.items(), key=lambda x: -x[1])[:3]:
                lines.append(f"            {lbl}: {p:.1%}")
        except FileNotFoundError:
            lines.append("PLS-DA  =>  model not trained yet")

        self.result_var.set("\n".join(lines))

        # Show a small preview of the uploaded image
        try:
            img    = Image.open(path).resize((180, 140))
            tk_img = ImageTk.PhotoImage(img)
            self.image_label.config(image=tk_img)
            self.image_label.image = tk_img     # keep reference to avoid garbage collection
        except Exception:
            pass

    # ── VIEW DATABASE SUMMARY ─────────────────────────────────────────────
    def view_summary(self):
        rows = self.db.fetch_all_labelled()
        if not rows:
            self.result_var.set("Database is empty.  Insert images first.")
            return
        from collections import Counter
        counts = Counter(lbl for _, lbl in rows)
        lines  = [f"Database: {len(rows)} images total", "-" * 38]
        for cls, cnt in sorted(counts.items()):
            bar = chr(9608) * min(cnt, 20)
            lines.append(f"{cls:25s} {cnt:3d}  {bar}")
        self.result_var.set("\n".join(lines))

    def __del__(self):
        try:
            self.db.close()
        except Exception:
            pass