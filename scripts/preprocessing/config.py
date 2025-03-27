"""
Configuration file for the complete cell analysis pipeline.
It consolidates settings for:
  1. Nuclear Segmentation (using StarDist)
  2. Cell Tracking and Analysis
  3. Apoptosis Annotation Matching and Evaluation
  4. Apoptosis Window Cropping and Dataset Generation
"""

from pathlib import Path

# Global settings
RUN_NAME = "test"

# Base directories
BASE_DATA_DIR = Path("../data") / RUN_NAME
BASE_LOG_DIR = Path("./logs") / RUN_NAME

# Global directories (common to several modules)
IMG_DIR = Path("/mnt/imaging.data/PertzLab/apoDetection/TIFFs")
MASK_DIR = BASE_DATA_DIR / "apo_masks"
DF_DIR = BASE_DATA_DIR / "summary_dfs"
DETAILS_DIR = BASE_DATA_DIR / "details"
TRACKED_MASK_DIR = BASE_DATA_DIR / "tracked_masks"
TRACK_DF_DIR = BASE_DATA_DIR / "track_dfs"
MATCH_CSV_DIR = BASE_DATA_DIR / "apo_match_csv"
PLOT_DIR = BASE_DATA_DIR / "plots"
LOG_DIR = BASE_LOG_DIR





# ---------------------------
# 1. Nuclear Segmentation
# ---------------------------
SEGMENTATION_CONFIG = {
    'IMG_DIR': IMG_DIR,                             # Input directory for images
    'MASK_DIR': MASK_DIR,             # Output: Segmentation masks
    'DF_DIR': DF_DIR,             # Output: Summary dataframes
    'DETAILS_DIR': DETAILS_DIR,            # Output: Detailed segmentation results
    'LOG_DIR': LOG_DIR,                            # Logging directory
    'SAVE_DATA': True,                              # Whether to save outputs
    'USE_GPU': True,                                # Enable GPU acceleration if available
    'MIN_NUC_SIZE': 200,                            # Minimum nuclear size to consider [pixels]
}

# ---------------------------
# 2. Cell Tracking and Analysis
# ---------------------------
TRACKING_CONFIG = {
    'IMG_DIR': IMG_DIR,                             # Input images directory (if needed)
    'MASK_DIR': MASK_DIR,                # Input segmentation masks (from segmentation pipeline)
    'DF_DIR': DF_DIR,                # Input summary dataframes from segmentation
    'TRACKED_MASK_DIR': TRACKED_MASK_DIR,     # Output: Masks with track IDs
    'TRACK_DF_DIR': TRACK_DF_DIR,            # Output: Track dataframes
    'PLOT_DIR': PLOT_DIR,                           # Directory to save plots (track length histograms, etc.)
    'RUN_NAME': RUN_NAME,                             # Name identifier for this run
    'BT_CONFIG_FILE': "extras/cell_config.json",    # Config file for the BTrack algorithm
    'EPS_TRACK': 70,                                # Tracking radius in pixels
    'TRK_MIN_LEN': 25,                             # Minimum track length in frames
    'LOG_DIR': LOG_DIR,
}

# ---------------------------
# 3. Apoptosis Annotation Matching and Evaluation
# ---------------------------
APO_MATCH_CONFIG = {
    'IMG_DIR': IMG_DIR,                             # Used for filename resolution
    'APO_DIR': '/mnt/imaging.data/PertzLab/apoDetection/ApoptosisAnnotation',  # Manual annotations directory
    'DETAILS_DIR': DETAILS_DIR,          # Input detailed segmentation results
    'MASK_DIR': MASK_DIR,           # Input segmentation masks (possibly modified for matching)
    'TRACKED_MASK_DIR': TRACKED_MASK_DIR,     # Input tracked masks
    'CSV_DIR': MATCH_CSV_DIR,        # Output CSV combining manual and automated detections
    'PLOT_DIR': PLOT_DIR,                           # Directory to save distance histograms
    'RUN_NAME': RUN_NAME,                          # Run identifier for matching evaluation
    'LOG_DIR':  LOG_DIR,
}

# ---------------------------
# 4. Apoptosis Window Cropping and Dataset Generation
# ---------------------------
APO_CROP_CONFIG = {
    'IMG_DIR': IMG_DIR,                             # Input images directory
    'EXPERIMENT_INFO': '/mnt/imaging.data/PertzLab/apoDetection/List of the experiments.csv',  # Experiment metadata
    'CSV_DIR': MATCH_CSV_DIR,         # Input CSV from annotation matching
    'TRACKED_MASK_DIR': TRACKED_MASK_DIR,       # Input tracked masks
    'TRACK_DF_DIR': TRACK_DF_DIR,              # Input tracking dataframes
    'CROPS_DIR': BASE_DATA_DIR / 'apo_crops',              # Output: Cropped image series for QC (.tif)
    'WINDOWS_DIR': BASE_DATA_DIR / 'windows',  # Output: Crops for machine learning (e.g., scDINO)
    'PLOT_DIR': PLOT_DIR,                           # Directory to save survival time histograms, etc.
    'RUN_NAME': RUN_NAME,                             # Run identifier for cropping
    'TRK_MIN_LEN': 25,                              # Minimum track length in frames (used for filtering)
    'MAX_TRACKING_DURATION': 20,                    # Maximum tracking duration in minutes
    'FRAME_INTERVAL': 5,                            # Temporal resolution between frames (in minutes)
    'WINDOW_SIZE': 48,                              # Size of spatial crops (pixels)
    'LOG_DIR': LOG_DIR,
}