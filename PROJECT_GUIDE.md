# Forensic Microbial Classification System: Project Guide

*Personal tracking document and PhD interview prep.*

**Working title (from abstract):** Qualitative Discrimination and Classification of Forensic Microbial Evidence Using Machine Learning in a Database-Integrated Framework
**Affiliation:** National Forensic Sciences University (NFSU), Delhi Campus
**Stack:** Python, scikit-image, scikit-learn, MySQL, Tkinter (MATLAB mentioned in the abstract for the PLS-DA/SVM comparison)

---

## 1. The 30-second pitch

> Microbial evidence at a crime scene is usually analysed by culture and morphological keys, which is slow, subjective, and varies between analysts. Genomics-based ML would be better but needs NGS infrastructure that many labs in India don't have. I built a lightweight pipeline that takes microscopy images, extracts interpretable morphological features (shape, texture, colony statistics), stores everything in a MySQL database, and classifies unknown samples with two classical models, SVM and PLS-DA, through a desktop GUI. It runs on a normal laptop and works with small datasets.

---

## 2. The problem and why it matters

| Issue with current practice | How this project responds |
|---|---|
| Culture/morphology-based identification is slow | Automated feature extraction and instant prediction |
| Subjective, with inter-analyst variability | Same feature pipeline and same model give reproducible output |
| Genomic ML needs NGS (scarce, costly) | Needs only compound-microscope images |
| Deep learning needs large datasets and GPUs | Classical ML works on small datasets and is easier to explain |
| Evidence data scattered in files | Central MySQL store used as both reference and training data |

**Forensic applications** (from the abstract): microbial profiling in bioterrorism and environmental crime investigations, and post-mortem interval estimation, especially where DNA is degraded or missing.

---

## 3. Classes predicted

Five morphological classes, with the subfolder name used as the label:

1. `gram_pos_cocci`
2. `gram_neg_bacilli`
3. `gram_pos_bacilli`
4. `gram_neg_cocci`
5. `fungi`

This is a **class-level (morphological)** discrimination, not species-level identification. Be precise about this in the interview.

---

## 4. System architecture

```
 Microscopy images (.png, one folder per class)
            │
            ▼
    augment.py  ──►  8x images per original (7 augmented + original)
            │
            ▼
    load_data.py ──►  MySQL: forensic_samples (image_blob, species, inserted_at)
            │
            ▼
   feature_extractor.py  ──►  1-D feature vector per image
            │
     ┌──────┴────────┐
     ▼               ▼
 svm_classifier   plsda_classifier
 (RBF SVM +       (PLS regression on
  grid search)     one-hot Y)
     │               │
     └──────┬────────┘
            ▼
   model/*.pkl (joblib)  ──►  MainWindow.py (Tkinter GUI) ──► prediction + confidence
```

---

## 5. File-by-file walkthrough

### `augment.py`: data augmentation (run once)
- Walks `data/<class>/` and takes each original `.png`, skipping files with `_aug_` in the name so it never augments its own output.
- Creates **7 variants** per original: rotations (90°, 180°, 270°), horizontal flip, vertical flip, darker (brightness ×0.8), brighter (×1.2).
- Result: 20 originals per class become 160 images per class.
- **Why:** small microscopy datasets are hard to collect. Microscopy images have no canonical orientation and lighting varies between slides, so these transforms are label-preserving and realistic.

### `load_data.py`: database ingestion (run once)
- Connects to MySQL, creates the table if missing, then loops through every class folder and inserts each image **as a binary blob** with the folder name as its label.
- Storing blobs in the DB keeps images and labels together in a single queryable reference source, which supports the "database-integrated" part of the title.

### `DataBaseClass.py`: database layer
- `Database` class wrapping `mysql.connector`.
- Table `forensic_samples(id, image_blob LONGBLOB, species VARCHAR(100), inserted_at TIMESTAMP)`.
- Methods: `connect`, `reconnect`, `create_table`, `insert_photo`, `fetch_all_labelled` (returns `(bytes, label)` tuples), `close`.
- The `inserted_at` timestamp gives a simple audit trail, which matters in forensic work.

### `feature_extractor.py`: the core of the method
Takes a file path, raw bytes (from MySQL) or a PIL image. It converts to grayscale, resizes to **128×128**, and builds a single vector from three descriptor families:

| Descriptor | Settings | What it captures | Size |
|---|---|---|---|
| **HOG** (Histogram of Oriented Gradients) | 9 orientations, 8×8 cells, 2×2 blocks | Edge/gradient structure, hence **shape** (cocci round vs bacilli rod-shaped) | 8,100 |
| **LBP** (Local Binary Pattern, uniform) | P=24, R=3, 26-bin normalised histogram | Local **texture** of colonies/cells | 26 |
| **Morphological stats** | Threshold at mean intensity, connected components, `regionprops` | Mean and std of object **area**, mean and std of **eccentricity**, object **count** | 5 |

**Total feature vector ≈ 8,131 dimensions.** (HOG: 15×15 blocks × 36 values = 8,100.)

Why this design is defensible: the features correspond to what a microbiologist looks at (shape, arrangement, texture, size), so the model's decisions can be discussed in domain terms.

### `svm_classifier.py`: Model 1, Support Vector Machine
- Labels are encoded with `LabelEncoder`.
- **80/20 stratified train/test split**.
- Pipeline: `StandardScaler` then `SVC(kernel="rbf", probability=True, class_weight="balanced")`.
- **Grid search** with 5-fold stratified CV over `C ∈ {0.1, 1, 10, 100}` and `gamma ∈ {scale, auto, 0.001, 0.01}`.
- Reports test accuracy, confusion matrix and classification report. Saves model and label encoder with `joblib`.
- `predict()` loads the model and returns `(class, confidence, {class: probability})`.
- **Why SVM:** it works well in high-dimensional, small-sample settings (here ~8k features), and the RBF kernel handles non-linear class boundaries.

### `plsda_classifier.py`: Model 2, PLS-DA
- Custom `PLSDAClassifier`: standardises X, **one-hot encodes** Y with `LabelBinarizer`, fits `PLSRegression`, and predicts with `argmax` over the predicted Y scores. This is the standard chemometric way to do PLS-DA, equivalent to `plsregress` in MATLAB.
- Components: `min(3, n_classes − 1)` (the rule of thumb that components ≤ classes − 1).
- Validation: **5-fold stratified CV**, refitting a fresh model per fold.
- `predict_proba` applies a **softmax** over the raw PLS scores so the GUI shows a readable confidence spread.
- **Why PLS-DA:** it is established in forensic chemistry and spectroscopy, handles correlated features (HOG features are highly collinear) by projecting to latent variables, and is interpretable via loadings and scores.

### `MainWindow.py`: Tkinter GUI
Dark-themed desktop app with five actions:
1. **Insert Labelled Image(s)**: pick images, choose a class in a popup, and they go straight into MySQL (lets the reference database grow over time).
2. **Train SVM Model**: pulls all rows from MySQL, extracts features, trains, shows accuracy.
3. **Train PLS-DA Model**: same, with CV accuracy.
4. **Classify Unknown Image**: extracts features from a new image and shows **both** models' predictions with top-3 class probabilities, plus an image preview.
5. **View Database Summary**: counts per class with a text bar chart.

A minimum of 6 labelled images is enforced before training.

### `Hackathon.py`: entry point
Creates the Tk root and launches `MainWindow`. Run with `python Hackathon.py`.

### `count_images.py`: dataset sanity checker
Counts PNGs per class, the total, and a 10% test-set size, and checks for stray images in the root folder.

---

## 6. End-to-end usage

```bash
python augment.py      # 1. expand dataset (once)
python load_data.py    # 2. load images + labels into MySQL (once)
python Hackathon.py    # 3. launch GUI → Train SVM → Train PLS-DA → Classify
```

---

## 7. Design decisions to be ready to defend

| Decision | Rationale |
|---|---|
| Classical ML instead of deep learning | Small dataset, no GPU needed, interpretable, suited to resource-limited labs |
| Hand-crafted features | Map onto morphological criteria used by microbiologists |
| Two different models | SVM is a strong non-linear classifier; PLS-DA is the chemometrics/forensic standard. Agreement between them increases trust, disagreement flags uncertain samples |
| MySQL | Persistent, queryable, extendable reference database, with the possibility of adding metadata (case ID, stain, magnification, date) |
| Stratified splits and CV | Keeps class proportions stable |
| Pipeline with scaler inside | Scaler is fit only on training folds during CV, so no preprocessing leakage |
| Extensible design | The same pipeline can take pollen, diatoms, hair, or fibre images by swapping data and labels |

---

## 8. Honest limitations and how to answer them

Supervisors respect candidates who find these problems themselves, so raise them first.

### 8.1 Augmentation leakage (the most important one)
**Issue:** images are augmented *before* going into the database, and the train/test split is then done randomly over all rows. A rotated copy of an image can land in the test set while the original is in training, which **inflates test accuracy**.
**Fix:** split by *original image ID* (group-aware split, e.g. `GroupShuffleSplit` / `StratifiedGroupKFold`), and augment only the training portion. Also report results on a truly held-out set (e.g. images from different slides, days, or microscopes).

### 8.2 PLS-DA evaluation is weaker than SVM's
- The classification report and confusion matrix for PLS-DA are computed on the **training data**, so they are optimistic. Only the CV number is a fair estimate.
- The two models are evaluated differently (SVM: 20% hold-out; PLS-DA: 5-fold CV), so they are not directly comparable.
**Fix:** use the same split and CV scheme for both, and report mean ± SD.

### 8.3 PLS-DA "probabilities" are not calibrated
Softmax over PLS regression outputs gives a readable score, not a true probability. **Answer:** "I treat it as a relative confidence score. Proper calibration would need Platt scaling or isotonic regression, or a permutation test for significance."

### 8.4 Dataset size and diversity
Roughly 20 originals per class (augmented to 160) is small, and augmented copies are not independent samples. **Answer:** this is a proof of concept. Next steps are more independent slides, multiple staining batches, multiple microscopes, and external validation.

### 8.5 Class-level only
The system separates morphological groups (e.g. Gram-positive cocci vs Gram-negative bacilli), not species. Gram status is not visible in grayscale unless stained images are used, so **colour information is currently discarded** (the image is converted to grayscale). Using colour features (e.g. HSV histograms) would likely help Gram-positive vs Gram-negative separation.

### 8.6 Crude segmentation
The "colony count/area/eccentricity" features use a mean-intensity threshold, which is sensitive to uneven illumination and debris. Better options: Otsu or adaptive thresholding, watershed for touching cells.

### 8.7 MATLAB component
The abstract mentions MATLAB and Python implementations, but this codebase is **Python only**. Be clear about what exists in the repository versus what is planned or done elsewhere.

### 8.8 Engineering and security
- The MySQL password is **hardcoded** in `load_data.py` and `MainWindow.py`. Move it to an environment variable or config file, and rotate the password if the code has been shared or pushed anywhere.
- No explicit duplicate-image detection, no user authentication, and no chain-of-custody features. These would be needed for a courtroom-grade system.
- `count_images.py` has a hardcoded Windows path and `.png`-only handling; GUI insertion accepts jpg/bmp too.

### 8.9 Forensic admissibility
Courts need known error rates, validation, and reproducibility (Daubert-style standards). The honest framing: this is a **screening and decision-support tool**, not a replacement for expert judgement.

---

## 9. Future work (a strong PhD proposal angle)

1. Group-aware validation and an external test set.
2. Colour and stain-aware features; better segmentation (Otsu, watershed).
3. Compare against a small CNN or transfer learning (ResNet/EfficientNet) with Grad-CAM, to test whether deep features beat hand-crafted ones on small data.
4. Add **open-set / "unknown" detection** so unseen organisms are flagged instead of forced into a known class.
5. Calibrated confidence and likelihood-ratio style reporting suitable for forensic evidence.
6. Extend to other trace evidence: pollen, diatoms, hair, fibres.
7. Link morphology to metadata and genomic data for post-mortem interval and environmental forensics.
8. Feature importance and PLS-DA loading/VIP analysis to show *which* features drive each class.

---

## 10. Likely interview questions and model answers

**Q1. What did you actually build?**
An end-to-end pipeline: augmentation, MySQL storage, feature extraction (HOG + LBP + morphology), two classifiers (SVM and PLS-DA), and a GUI for training and classifying unknown images.

**Q2. Why not deep learning?**
Small dataset, no GPU, and forensic users need explainability. Classical ML on interpretable features is more suitable here. I'd like to benchmark a CNN as future work and let the data decide.

**Q3. Why those three feature types?**
HOG captures shape (round vs rod), LBP captures texture, and the morphological stats capture size, elongation and density, which are the cues microbiologists use.

**Q4. Why SVM and PLS-DA together?**
SVM is a strong non-linear classifier for high-dimensional small samples. PLS-DA is the chemometrics standard for correlated features and is interpretable. Using both gives a cross-check.

**Q5. What accuracy did you get, and do you trust it?**
Quote your actual numbers, then add: *"I want to be upfront that augmented copies could appear in both train and test splits, so I treat this as an optimistic estimate. I've identified a group-aware split as the fix, and an independent validation set is my next step."*

**Q6. How many features, and isn't that too many for the sample size?**
About 8,100 dimensions, mostly HOG. That is why I use SVM, which is robust in high dimensions, and PLS-DA, which compresses to 3 latent variables. Dimensionality reduction or feature selection is a natural improvement.

**Q7. Why n_components = 3 in PLS-DA?**
With 5 classes the rule of thumb is components ≤ classes − 1, capped at 3 here. I would tune it properly by cross-validated Q² / minimum CV error.

**Q8. What would make this court-ready?**
Larger validated datasets, known error rates, calibrated confidence, an open-set rejection option, audit logging and chain of custody, and standardised imaging protocols.

**Q9. How would you extend it to other evidence types?**
Swap the dataset and labels, adjust feature extractors if needed (e.g. fibre texture, diatom shape), and reuse the database, training and GUI layers unchanged.

**Q10. What would you do in your PhD with this?**
Build a validated, multi-evidence, open-set morphological classification framework with calibrated, courtroom-appropriate uncertainty reporting, compare classical and deep approaches, and test on real casework-like samples.

---

## 11. Quick reference: key numbers and terms

| Item | Value |
|---|---|
| Image size after resize | 128×128 grayscale |
| HOG | 9 orientations, 8×8 cell, 2×2 block (8,100 features) |
| LBP | P=24, R=3, uniform, 26 bins |
| Morphological features | 5 |
| Total features | ≈ 8,131 |
| Augmentations per image | 7 (3 rotations, 2 flips, dark, bright) |
| SVM | RBF, balanced class weights, grid over C and gamma, 5-fold CV, 80/20 split |
| PLS-DA | min(3, K−1) components, StandardScaler, softmax scores, 5-fold CV |
| Classes | 5 |
| Min images to train | 6 |

**Terms to be fluent in:** HOG, LBP, RBF kernel, C and gamma, stratified CV, PLS-DA latent variables, VIP scores, data leakage, calibration, open-set recognition, Daubert standard.

---

## 12. Running checklist (update as you go)

- [ ] Add your real accuracy/CV numbers to Section 10, Q5
- [ ] Implement group-aware train/test split
- [ ] Report SVM and PLS-DA with identical evaluation
- [ ] Move DB password to environment variable
- [ ] Add confusion-matrix figures and PLS-DA score plot for slides or interview
- [ ] Add colour features and better segmentation
- [ ] Clarify the MATLAB part (what is done vs planned)
- [ ] Prepare a 2-minute live demo (insert → train → classify)
