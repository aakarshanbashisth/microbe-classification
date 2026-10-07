import numpy as np
from PIL import Image
from skimage.feature import hog, local_binary_pattern
from skimage.measure import regionprops, label as sk_label
import io

IMG_SIZE = (128, 128)

def extract_features(image_input):
    """
    Convert any microscopy image into a feature vector for ML.

    Accepts:
        image_input  — file path (str), raw bytes from MySQL, or PIL Image

    Returns:
        1-D numpy array  (HOG + LBP + shape descriptors concatenated)
    """
    # ── Load image ──────────────────────────────────────────────────────────
    if isinstance(image_input, bytes):
        img = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, str):
        img = Image.open(image_input)
    else:
        img = image_input

    # Convert to grayscale and resize to fixed 128×128
    gray = np.array(img.convert("L").resize(IMG_SIZE))

    # ── 1. HOG — detects edges and gradient structure (shape) ───────────────
    hog_feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True
    )

    # ── 2. LBP — detects surface texture (excellent for microbial colonies) ─
    lbp          = local_binary_pattern(gray, P=24, R=3, method="uniform")
    lbp_hist, _  = np.histogram(lbp.ravel(), bins=26,
                                range=(0, 26), density=True)

    # ── 3. Morphological descriptors — colony size, eccentricity, count ─────
    binary  = gray < gray.mean()
    labeled = sk_label(binary)
    props   = regionprops(labeled)
    if props:
        areas      = [r.area        for r in props]
        eccentrics = [r.eccentricity for r in props]
        shape_feat = np.array([
            np.mean(areas), np.std(areas),
            np.mean(eccentrics), np.std(eccentrics),
            float(len(props))           # number of detected colonies/cells
        ])
    else:
        shape_feat = np.zeros(5)

    # Concatenate all three into one long vector
    return np.concatenate([hog_feat, lbp_hist, shape_feat])