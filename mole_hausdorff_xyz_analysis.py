"""
mole_hausdorff_xyz_analysis.py

Full pipeline for skin mole analysis using CIE XYZ color space:

1. Segments moles using the Y (luminance) channel + Otsu thresholding
2. Saves cropped & centered 300x300 segmented mole images
3. Computes Hausdorff (fractal) dimension for:
   - Whole mole region
   - Border only
4. Extracts XYZ color statistics (mean, median, std, min, max) from:
   - Whole mole (default) OR
   - Border pixels only (toggleable)
5. Performs statistical comparison between benign and malignant

"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage import io
from scipy.ndimage import binary_fill_holes
from scipy.stats import shapiro, ttest_ind, mannwhitneyu


# 1. Mole Segmentation (XYZ → Y channel)
def segment_mole_xyz(image_path, save_path=None, visualize=False):
    """Segment mole using XYZ color space and Otsu on Y channel."""
    image = io.imread(image_path)  # RGB

    xyz_image = cv2.cvtColor(image, cv2.COLOR_RGB2XYZ)
    Y_channel = xyz_image[:, :, 1]

    blurred = cv2.GaussianBlur(Y_channel, (5, 5), 0)
    _, thresholded = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, None

    largest_contour = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(Y_channel)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    mask = binary_fill_holes(mask).astype(np.uint8) * 255

    segmented = cv2.bitwise_and(image, image, mask=mask)

    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped = segmented[y : y + h, x : x + w]
    cropped_bgr = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
    cropped_resized = cv2.resize(cropped_bgr, (300, 300))

    if save_path:
        cv2.imwrite(save_path, cropped_resized)

    if visualize:
        titles = [
            "Original",
            "Y Channel",
            "Blurred",
            "Thresholded",
            "Mask",
            "Segmented",
        ]
        images = [image, Y_channel, blurred, thresholded, mask, segmented]
        plt.figure(figsize=(15, 10))
        for i, img in enumerate(images):
            plt.subplot(2, 3, i + 1)
            if len(img.shape) == 2:
                plt.imshow(img, cmap="gray")
            else:
                plt.imshow(img[..., ::-1])
            plt.title(titles[i])
            plt.axis("off")
        plt.tight_layout()
        plt.show()

    return mask, (cropped_resized if not save_path else None)


# 2. Hausdorff Dimension (Box-Counting)
def hausdorff_dimension(binary_image):
    """Box-counting fractal dimension estimation."""
    if binary_image.sum() == 0:
        return np.nan

    max_dim = max(binary_image.shape)
    new_size = 2 ** int(np.ceil(np.log2(max_dim)))
    pad_h = new_size - binary_image.shape[0]
    pad_w = new_size - binary_image.shape[1]
    padded = np.pad(binary_image, ((0, pad_h), (0, pad_w)), mode="constant")

    box_counts = []
    resolutions = []
    box_size = padded.shape[0]

    while box_size >= 1:
        boxes_per_side = padded.shape[0] // box_size
        count = 0
        for i in range(boxes_per_side):
            for j in range(boxes_per_side):
                patch = padded[
                    i * box_size : (i + 1) * box_size,
                    j * box_size : (j + 1) * box_size,
                ]
                if np.any(patch):
                    count += 1
        box_counts.append(count)
        resolutions.append(1 / box_size)
        box_size //= 2

    if len(box_counts) < 2:
        return np.nan

    coeffs = np.polyfit(np.log(resolutions), np.log(box_counts), 1)
    return coeffs[0]


# 3. Batch Segmentation + Hausdorff
def process_folders_for_hausdorff(
    benign_dir, malignant_dir, segmented_benign_dir, segmented_malignant_dir
):
    os.makedirs(segmented_benign_dir, exist_ok=True)
    os.makedirs(segmented_malignant_dir, exist_ok=True)

    results = []

    for filename in tqdm(
        os.listdir(benign_dir), desc="Benign (XYZ Segmentation)"
    ):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            src = os.path.join(benign_dir, filename)
            save = os.path.join(segmented_benign_dir, filename)
            mask, _ = segment_mole_xyz(src, save_path=save)
            if mask is None:
                continue

            mole_dim = hausdorff_dimension(mask.astype(bool))

            border_mask = np.zeros_like(mask)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(border_mask, contours, -1, 255, thickness=1)
            border_dim = hausdorff_dimension(border_mask.astype(bool))

            results.append(
                {
                    "Image": filename,
                    "Type": "Benign",
                    "Mole_Hausdorff_Dim": mole_dim,
                    "Border_Hausdorff_Dim": border_dim,
                }
            )

    for filename in tqdm(
        os.listdir(malignant_dir), desc="Malignant (XYZ Segmentation)"
    ):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            src = os.path.join(malignant_dir, filename)
            save = os.path.join(segmented_malignant_dir, filename)
            mask, _ = segment_mole_xyz(src, save_path=save)
            if mask is None:
                continue

            mole_dim = hausdorff_dimension(mask.astype(bool))
            border_mask = np.zeros_like(mask)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(border_mask, contours, -1, 255, thickness=1)
            border_dim = hausdorff_dimension(border_mask.astype(bool))

            results.append(
                {
                    "Image": filename,
                    "Type": "Malignant",
                    "Mole_Hausdorff_Dim": mole_dim,
                    "Border_Hausdorff_Dim": border_dim,
                }
            )

    df = pd.DataFrame(results)
    df.to_csv("hausdorff_dimensions_xyz.csv", index=False)
    print("Hausdorff dimensions saved → hausdorff_dimensions_xyz.csv")
    return df


# 4. XYZ Feature Extraction with Border Toggle
def extract_xyz_features(
    segmented_dir, label, border_only=False, border_thickness=8
):
    """
    Extract XYZ statistics.

    Parameters:
        border_only (bool): True: extract only from border ring
        border_thickness (int): Width of border ring in pixels (used only if border_only=True)
    """
    features = []
    for filename in tqdm(
        os.listdir(segmented_dir),
        desc=f"XYZ Features ({label} - {'Border' if border_only else 'Whole'})",
    ):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(segmented_dir, filename)
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            continue

        img_xyz = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2XYZ)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Basic mole mask (exclude pure black background)
        _, mole_mask = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY)

        if border_only:
            # Create thick border ring
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (border_thickness, border_thickness)
            )
            dilated = cv2.dilate(mole_mask, kernel, iterations=1)
            eroded = cv2.erode(mole_mask, kernel, iterations=1)
            border_mask = dilated - eroded
            valid_mask = border_mask
        else:
            valid_mask = mole_mask

        valid_pixels = img_xyz[valid_mask > 0]
        if valid_pixels.size == 0:
            print(
                f"Warning: No valid pixels in {filename} (border_only={border_only})"
            )
            continue

        X, Y, Z = valid_pixels[:, 0], valid_pixels[:, 1], valid_pixels[:, 2]

        features.append(
            {
                "filename": filename,
                "label": label,
                "region": "border" if border_only else "whole",
                "mean_X": np.mean(X),
                "median_X": np.median(X),
                "std_X": np.std(X),
                "min_X": np.min(X),
                "max_X": np.max(X),
                "mean_Y": np.mean(Y),
                "median_Y": np.median(Y),
                "std_Y": np.std(Y),
                "min_Y": np.min(Y),
                "max_Y": np.max(Y),
                "mean_Z": np.mean(Z),
                "median_Z": np.median(Z),
                "std_Z": np.std(Z),
                "min_Z": np.min(Z),
                "max_Z": np.max(Z),
            }
        )

    return features


# 5. Statistical Comparison
def statistical_comparison_xyz(features_df):
    features = ["mean", "median", "std", "min", "max"]
    channels = ["X", "Y", "Z"]
    results = []

    for channel in channels:
        for feat in features:
            col = f"{feat}_{channel}"
            benign = features_df[features_df["label"] == "benign"][col].dropna()
            malignant = features_df[features_df["label"] == "malignant"][
                col
            ].dropna()

            if len(benign) < 3 or len(malignant) < 3:
                continue

            p_benign = shapiro(benign).pvalue
            p_mal = shapiro(malignant).pvalue

            if p_benign > 0.05 and p_mal > 0.05:
                stat, p_val = ttest_ind(benign, malignant)
                test = "t-test"
            else:
                stat, p_val = mannwhitneyu(benign, malignant)
                test = "Mann-Whitney U"

            results.append(
                {
                    "Feature": col,
                    "Test Type": test,
                    "Statistic": stat,
                    "P-value": p_val,
                }
            )

    results_df = pd.DataFrame(results)
    results_df.to_csv("xyz_statistical_results.csv", index=False)

    print("\n=== Significant Features (p < 0.05) ===")
    print(results_df[results_df["P-value"] < 0.05])
    print("\n=== All Results ===")
    print(results_df)


# 6. Main Execution

if __name__ == "__main__":
    # UPDATE
    RAW_BENIGN_DIR = r"..."
    RAW_MALIGNANT_DIR = r"..."
    SEGMENTED_BENIGN_DIR = r"..."
    SEGMENTED_MALIGNANT_DIR = r"..."

    # Choose what region to analyze for color features
    EXTRACT_BORDER_ONLY = False  # Set to True for border-only analysis
    BORDER_THICKNESS = 8  # Only used if EXTRACT_BORDER_ONLY=True

    region_name = "BORDER ONLY" if EXTRACT_BORDER_ONLY else "WHOLE MOLE"
    print(f"\n=== Mole Analysis Pipeline (XYZ) – {region_name} ===")

    print("\n1. Segmenting moles and computing Hausdorff dimensions...")
    process_folders_for_hausdorff(
        RAW_BENIGN_DIR,
        RAW_MALIGNANT_DIR,
        SEGMENTED_BENIGN_DIR,
        SEGMENTED_MALIGNANT_DIR,
    )

    print(f"\n2. Extracting XYZ features ({region_name})...")
    benign_feats = extract_xyz_features(
        SEGMENTED_BENIGN_DIR,
        "benign",
        border_only=EXTRACT_BORDER_ONLY,
        border_thickness=BORDER_THICKNESS,
    )
    malignant_feats = extract_xyz_features(
        SEGMENTED_MALIGNANT_DIR,
        "malignant",
        border_only=EXTRACT_BORDER_ONLY,
        border_thickness=BORDER_THICKNESS,
    )

    all_features = benign_feats + malignant_feats
    if not all_features:
        print("ERROR: No features extracted!")
    else:
        df = pd.DataFrame(all_features)
        suffix = "_border" if EXTRACT_BORDER_ONLY else "_whole"
        feature_csv = f"mole_xyz_features{suffix}.csv"
        df.to_csv(feature_csv, index=False)
        print(f"✓ Features saved → {feature_csv} ({len(all_features)} samples)")

        print("\n3. Statistical comparison (benign vs malignant)...")
        statistical_comparison_xyz(df)

    print("Outputs:")
    print("   • hausdorff_dimensions_xyz.csv")
    print(f"   • {feature_csv}")
    print("   • xyz_statistical_results.csv")
    print(f"   • Segmented images in XYZ folders")
