# Skin-Mole-Analysis---Border-and-Color
Four files including code for analyzing images of skin moles. (Segmentation, calculating Box counting dimension, converting to Lab, XYZ, HSV and YCbCr color space, analysis of the different channels and statistical analysis of the acquired results).

The hausDim function is written by Alceu Ferraz Costa. It is converted to python for this analysis. 
## Features
Four independent Python scripts:

1. mole_hausdorff_hsv_analysis.py <br>
2. mole_hausdorff_xyz_analysis.py <br>
3. mole_hausdorff_ycbcr_analysis.py <br>
4. mole_hausdorff_lab_analysis.py <br>

Each script performs the exact same workflow but in a different color space (HSV, CIE XYZ, YCbCr, or Lab). 

## What Each Script Does (Step-by-Step)<br>
When you run one of these scripts, it will automatically perform the following in sequence:

#### Segmentation<br>
- Reads all original images from the Benign and Malignant folders <br>
- Segments the mole using the luminance/lightness channel of the respective color space (V for HSV, Y for XYZ & YCbCr, L for Lab) + Otsu thresholding <br>
- Crops the mole tightly and resizes it to 300×300 pixels <br>
- Saves the segmented images into a new subfolder <br>


#### Hausdorff (Fractal) Dimension Calculation<br>
- Computes fractal dimension of the whole filled mole <br>
- Computes fractal dimension of the mole border only (thin contour)<br>
- Saves results to: hausdorff_dimensions_[colorspace].csv<br>

#### Color Feature Extraction <br>
- Extracts statistical features (mean, median, std, min, max) from all three channels <br>
- You can choose (via toggle) to extract from:<br>
        - Whole mole (default) <br>
        - Border ring only<br>
- Saves to: mole_[colorspace]_features_whole.csv or mole_[colorspace]_features_border.csv <br>

#### Statistical Comparison<br>
- Tests each feature for normality (Shapiro-Wilk)<br>
- Performs appropriate test (t-test if normal, Mann-Whitney U otherwise)<br>
- Prints and saves significant features (p < 0.05)<br>
- Saves full results to: [colorspace]_statistical_results.csv<br>


## Setup and Installation

### 1. Requirements
All dependencies are listed in `requirements.txt`.

```bash
# Create and activate a virtual environment 
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt


  


