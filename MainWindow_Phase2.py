# MainWindow_Phase2.py
# Demo GUI for FIU 2026 presentation
# Colour scheme matched to presentation slides — white + navy #1F3864

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import joblib, os
import numpy as np
from feature_extractor import extract_features
from species_map       import SPECIES_MAP

MODEL_DIR      = "C:\\AI-ML\\model"
EVIDENCE_TYPES = ["microbe", "diatom", "pollen"]

# ── Colour scheme — matched to presentation slides ──────────────
BG_WHITE    = "#FFFFFF"
BG_LIGHT    = "#F0F4FA"
BG_CARD     = "#F8FAFC"
NAVY        = "#000000"
NAVY_LIGHT  = "#303234"
NAVY_PALE   = "#E8EEF7"
FG_BLACK    = "#1A1A2E"
FG_MUTED    = "#6B7280"
FG_GREEN    = "#166534"
BG_GREEN    = "#DCFCE7"
BORDER      = "#CBD5E1"
ACCENT_SVM  = "#000000"
ACCENT_PLS  = "#303234"

def load_final_model(ev_type, model_name):
    if model_name == "SVM":
        m_path = os.path.join(MODEL_DIR, f"{ev_type}_svm_final.pkl")
        l_path = os.path.join(MODEL_DIR, f"{ev_type}_svm_labels_final.pkl")
        if not os.path.exists(m_path):
            raise FileNotFoundError(f"Run Step 6 first: {m_path}")
        return joblib.load(m_path), joblib.load(l_path)
    else:
        m_path = os.path.join(MODEL_DIR, f"{ev_type}_plsda_final.pkl")
        if not os.path.exists(m_path):
            raise FileNotFoundError(f"Run Step 6 first: {m_path}")
        return joblib.load(m_path), None


class Phase2App:
    def __init__(self, root):
        self.root = root
        self.root.title(
            "Forensic Evidence Classification System")
        self.root.geometry("920x740")
        self.root.configure(bg=BG_WHITE)
        self.root.resizable(True, True)

        self._build_header()
        self._build_controls()
        self._build_image_panel()
        self._build_results_panel()

    # ── HEADER ──────────────────────────────────────────────────
    def _build_header(self):
        hdr = tk.Frame(self.root, bg=NAVY)
        hdr.pack(fill="x", padx=0, pady=0)

        tk.Label(hdr,
                 text="Forensic Evidence Classification System",
                 font=("Georgia", 16, "bold"),
                 bg=NAVY, fg=BG_WHITE).pack(pady=(14, 3))
        tk.Label(hdr,
                 text="SVM + PLS-DA  ·  Microbes  ·  Diatoms  ·  Pollens " ,
                 font=("Calibri", 9),
                 bg=NAVY, fg="#A8BFDA").pack(pady=(0, 12))

    # ── CONTROLS ────────────────────────────────────────────────
    def _build_controls(self):
        ctrl_outer = tk.Frame(self.root, bg=NAVY_PALE,
                              highlightbackground=NAVY,
                              highlightthickness=1)
        ctrl_outer.pack(fill="x", padx=20, pady=(14, 0))

        ctrl = tk.Frame(ctrl_outer, bg=NAVY_PALE)
        ctrl.pack(fill="x", padx=16, pady=10)

        tk.Label(ctrl, text="Evidence Type:",
                 font=("Calibri", 10, "bold"),
                 bg=NAVY_PALE, fg=NAVY).grid(
                     row=0, column=0, sticky="w", padx=(0, 12))

        self.ev_var = tk.StringVar(value="microbe")
        for i, et in enumerate(EVIDENCE_TYPES):
            tk.Radiobutton(ctrl, text=et.capitalize(),
                           variable=self.ev_var, value=et,
                           font=("Calibri", 10),
                           bg=NAVY_PALE, fg=FG_BLACK,
                           selectcolor=BG_WHITE,
                           activebackground=NAVY_PALE,
                           command=self._clear_results).grid(
                               row=0, column=i + 1, padx=10)

        tk.Button(ctrl,
                  text="  Select Image & Classify  →",
                  command=self.classify,
                  font=("Calibri", 10, "bold"),
                  bg=NAVY, fg=BG_WHITE,
                  bd=0, padx=18, pady=7, cursor="hand2",
                  relief="flat",
                  activebackground=NAVY_LIGHT,
                  activeforeground=BG_WHITE).grid(
                      row=0, column=5, padx=(20, 0), sticky="e")

        ctrl.columnconfigure(5, weight=1)

    # ── IMAGE PANEL ─────────────────────────────────────────────
    def _build_image_panel(self):
        self.img_frame = tk.Frame(self.root, bg=BG_WHITE)
        self.img_frame.pack(fill="x", padx=20, pady=(12, 0))

        img_border = tk.Frame(self.img_frame, bg=NAVY,
                              highlightthickness=0, padx=2, pady=2)
        img_border.pack(side="left")

        self.img_label = tk.Label(img_border, bg=BG_LIGHT,
                                  width=24, height=10,
                                  text="No image selected",
                                  font=("Calibri", 9), fg=FG_MUTED)
        self.img_label.pack()

        info_frame = tk.Frame(self.img_frame, bg=BG_WHITE)
        info_frame.pack(side="left", fill="both", expand=True, padx=16)

        self.file_info = tk.Label(info_frame,
                                  text="Select an evidence image to begin classification.",
                                  font=("Courier", 10),
                                  bg=BG_WHITE, fg=FG_MUTED,
                                  justify="left", anchor="nw")
        self.file_info.pack(anchor="nw", pady=4)

    # ── RESULTS PANEL ────────────────────────────────────────────
    def _build_results_panel(self):
        tk.Frame(self.root, bg=BORDER, height=1).pack(
            fill="x", padx=20, pady=10)

        self.res_frame = tk.Frame(self.root, bg=BG_WHITE)
        self.res_frame.pack(fill="both", expand=True, padx=20, pady=(0, 16))

        self.svm_frame = self._make_model_panel(
            self.res_frame, "SVM", ACCENT_SVM)
        self.svm_frame.pack(side="left", fill="both", expand=True,
                            padx=(0, 10))

        self.pls_frame = self._make_model_panel(
            self.res_frame, "PLS-DA", ACCENT_PLS)
        self.pls_frame.pack(side="left", fill="both", expand=True)

    def _make_model_panel(self, parent, title, accent):
        outer = tk.Frame(parent, bg=BG_WHITE,
                         highlightbackground=accent,
                         highlightthickness=2)

        # Header bar
        hdr = tk.Frame(outer, bg=accent)
        hdr.pack(fill="x")
        tk.Label(hdr, text=title,
                 font=("Georgia", 12, "bold"),
                 bg=accent, fg=BG_WHITE,
                 pady=8).pack(side="left", padx=14)
        tk.Label(hdr, text="Classification Result",
                 font=("Calibri", 9),
                 bg=accent, fg="#A8BFDA",
                 pady=8).pack(side="right", padx=14)

        inner = tk.Frame(outer, bg=BG_WHITE)
        inner.pack(fill="both", expand=True, padx=14, pady=12)

        # Predicted species
        tk.Label(inner, text="PREDICTED SPECIES",
                 font=("Calibri", 8, "bold"),
                 bg=BG_WHITE, fg=FG_MUTED).pack(anchor="w")

        pred_lbl = tk.Label(inner, text="—",
                            font=("Georgia", 14, "bold italic"),
                            bg=BG_WHITE, fg=FG_BLACK,
                            wraplength=380, justify="left")
        pred_lbl.pack(anchor="w", pady=(2, 6))

        # Class label
        class_frame = tk.Frame(inner, bg=NAVY_PALE,
                               highlightbackground=NAVY,
                               highlightthickness=1)
        class_frame.pack(anchor="w", pady=(0, 6))

        class_lbl = tk.Label(class_frame, text="Class: —",
                             font=("Calibri", 10, "bold"),
                             bg=NAVY_PALE, fg=NAVY,
                             padx=10, pady=4)
        class_lbl.pack()

        # Confidence
        conf_lbl = tk.Label(inner, text="",
                            font=("Calibri", 11, "bold"),
                            bg=BG_WHITE, fg=FG_GREEN)
        conf_lbl.pack(anchor="w", pady=(0, 8))

        # Separator
        tk.Frame(inner, bg=BORDER, height=1).pack(fill="x", pady=(0, 8))

        # Label for chart
        tk.Label(inner, text="TOP SPECIES PROBABILITIES",
                 font=("Calibri", 8, "bold"),
                 bg=BG_WHITE, fg=FG_MUTED).pack(anchor="w")

        canvas = tk.Canvas(inner, bg=BG_WHITE, bd=0,
                           highlightthickness=0, height=190)
        canvas.pack(fill="both", expand=True, pady=(6, 0))

        setattr(self, f"_{title.replace('-','')}_pred",  pred_lbl)
        setattr(self, f"_{title.replace('-','')}_class", class_lbl)
        setattr(self, f"_{title.replace('-','')}_conf",  conf_lbl)
        setattr(self, f"_{title.replace('-','')}_canvas",canvas)

        return outer

    # ── CLASSIFY ────────────────────────────────────────────────
    def classify(self):
        path = filedialog.askopenfilename(
            title="Select forensic evidence image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return

        ev_type = self.ev_var.get()

        try:
            img    = Image.open(path).resize((200, 150))
            tk_img = ImageTk.PhotoImage(img)
            self.img_label.config(image=tk_img, text="")
            self.img_label.image = tk_img
        except Exception:
            pass

        fname = os.path.basename(path)
        self.file_info.config(
            text=f"File       :  {fname}\n"
                 f"Evidence   :  {ev_type.capitalize()}\n"
                 f"Status     :  Extracting morphological features...")
        self.root.update()

        try:
            feat = extract_features(path)
        except Exception as e:
            messagebox.showerror("Feature Extraction Error", str(e))
            return

        self.file_info.config(
            text=f"File       :  {fname}\n"
                 f"Evidence   :  {ev_type.capitalize()}\n"
                 f"Features   :  {len(feat)} dimensions (HOG + LBP + Morphology)\n"
                 f"Status     :  Classifying...")
        self.root.update()

        for model_name, attr_prefix in [("SVM", "SVM"), ("PLS-DA", "PLSDA")]:
            try:
                model, le = load_final_model(ev_type, model_name)
            except FileNotFoundError:
                messagebox.showerror("Model Not Found",
                    f"{model_name} final model not found.\n"
                    "Run p2_step6_train_final.py first.")
                return

            X = np.array(feat).reshape(1, -1)

            if model_name == "SVM":
                pred_enc  = model.predict(X)[0]
                proba     = model.predict_proba(X)[0]
                species   = le.inverse_transform([pred_enc])[0]
                prob_dict = dict(zip(le.classes_, proba))
            else:
                species   = model.predict(X)[0]
                proba     = model.predict_proba(X)[0]
                prob_dict = dict(zip(model.classes_, proba))

            cls_label, _ = SPECIES_MAP.get(species, ("unknown", ""))
            confidence   = float(max(prob_dict.values()))

            pred_w  = getattr(self, f"_{attr_prefix}_pred")
            class_w = getattr(self, f"_{attr_prefix}_class")
            conf_w  = getattr(self, f"_{attr_prefix}_conf")
            canvas  = getattr(self, f"_{attr_prefix}_canvas")

            pred_w.config(text=species.replace("_", " "))
            class_w.config(text=f"  Class: {cls_label.replace('_',' ').title()}  ")
            conf_lbl_text = f"Confidence: {confidence:.1%}"
            conf_w.config(text=conf_lbl_text,
                          fg=FG_GREEN if confidence >= 0.6 else "#92400E")

            self._draw_bars(canvas, prob_dict, species)

        self.file_info.config(
            text=f"File       :  {fname}\n"
                 f"Evidence   :  {ev_type.capitalize()}\n"
                 f"Features   :  {len(feat)} dimensions (HOG + LBP + Morphology)\n"
                 f"Status     :  Classification complete  ✓")

    def _draw_bars(self, canvas, prob_dict, top_species):
        canvas.delete("all")
        canvas.update_idletasks()
        W = canvas.winfo_width()  or 380
        H = canvas.winfo_height() or 190

        sorted_items = sorted(prob_dict.items(), key=lambda x: -x[1])[:6]
        n = len(sorted_items)
        if n == 0:
            return

        bar_h   = min(22, (H - 20) // n - 8)
        y_start = 8
        label_w = 168
        bar_max = W - label_w - 56
        gap     = bar_h + 10

        for i, (sp, prob) in enumerate(sorted_items):
            y     = y_start + i * gap
            bar_w = max(2, int(prob * bar_max))
            is_top = sp == top_species

            color    = NAVY     if is_top else "#CBD5E1"
            txt_col  = FG_BLACK if is_top else FG_MUTED
            val_col  = NAVY     if is_top else FG_MUTED

            sp_short = sp.replace("_", " ")
            if len(sp_short) > 24:
                sp_short = sp_short[:23] + "…"

            # Species name — bold if top
            canvas.create_text(
                label_w - 8, y + bar_h // 2,
                text=sp_short, anchor="e",
                font=("Calibri", 9, "bold" if is_top else "normal"),
                fill=txt_col)

            # Bar background
            canvas.create_rectangle(
                label_w, y, label_w + bar_max, y + bar_h,
                fill="#F1F5F9", outline=BORDER)

            # Bar fill
            canvas.create_rectangle(
                label_w, y, label_w + bar_w, y + bar_h,
                fill=color, outline="")

            # Percentage
            canvas.create_text(
                label_w + bar_w + 6, y + bar_h // 2,
                text=f"{prob:.1%}", anchor="w",
                font=("Calibri", 9, "bold" if is_top else "normal"),
                fill=val_col)

    def _clear_results(self):
        for attr_prefix in ["SVM", "PLSDA"]:
            getattr(self, f"_{attr_prefix}_pred" ).config(text="—")
            getattr(self, f"_{attr_prefix}_class").config(text="Class: —")
            getattr(self, f"_{attr_prefix}_conf" ).config(text="")
            getattr(self, f"_{attr_prefix}_canvas").delete("all")
        self.file_info.config(
            text="Select an evidence image to begin classification.")


if __name__ == "__main__":
    root = tk.Tk()
    app  = Phase2App(root)
    root.mainloop()