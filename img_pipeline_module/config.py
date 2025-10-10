"""
Configuration file for the complete cell analysis pipeline.
It consolidates settings for:
  1. Nuclear Segmentation (using StarDist)
  2. Cell Tracking and Analysis
  3. Apoptosis Annotation Matching and Evaluation
  4. Apoptosis Window Cropping and Dataset Generation
  5. Crop Upsampling
"""

from pathlib import Path

# ==================================
# GLOBAL & EXTERNAL SETTINGS
# ==================================
RUN_NAME = "new_config_test"

# 1. Base Project Directories (Relative to execution)
BASE_DATA_DIR = Path("./data") / RUN_NAME
BASE_LOG_DIR = Path("./logs") / RUN_NAME

# 2. External Absolute Paths (Must be manually updated for new environments)
EXTERNAL_PATHS = {
    'EXPERIMENT_INFO_CSV': '/mnt/imaging.data/PertzLab/apoDetection/List of the experiments.csv',
    'APO_ANNOTATIONS_DIR': '/mnt/imaging.data/PertzLab/apoDetection/ApoptosisAnnotation',
    'SOURCE_IMAGES_DIR': Path("/home/nbahou/myimaging/test_tiffs/mini"), 
}

# 3. Centralized Output Directory Names (Single Source of Truth)
# These are the *names* of the folders inside BASE_DATA_DIR
OUTPUT_DIRS = {
    'DETAILS': 'details',
    'TRACKED_MASKS': 'tracked_masks_tiff',
    'TRACK_DF': 'track_dfs',
    'APO_MATCH_CSV': 'apo_match_csv',
    'PLOTS': 'plots',

    # Segmentation Output Names
    'MASK_DIR': 'apo_masks',    # maybe better strdst_masks
    'MASK_DIR_NO_FILT': 'apo_masks_no_filt',

    
    # Cropping Output Names
    'WINDOW_CROPS_BASE': 'windows',
    'WINDOW_CROPS_20X_BASE': 'windows_20x',
    'WINDOW_CROPS_UPSAMPLED': 'windows_20x_2cat_resize_128',
    'REJECTED_CROPS': 'bad_crops',
    'CROPS_DIR': 'apo_crops',
    'APO_CHECK_ARRAYS': 'apo_check_arrays',
    'FILTER_FEATURES': 'features_df'
}

# ==================================
# 1. Nuclear Segmentation
# ==================================
SEGMENTATION_CONFIG = {
    # External Input
    'IMG_DIR': EXTERNAL_PATHS['SOURCE_IMAGES_DIR'], 
    'EXPERIMENT_INFO': EXTERNAL_PATHS['EXPERIMENT_INFO_CSV'],
    
    # Output Directories (if saving intermediate files)
    'MASK_DIR': BASE_DATA_DIR / 'apo_masks',
    'MASK_DIR_NO_FILT': BASE_DATA_DIR / 'apo_masks_no_filt',
    'DF_DIR': BASE_DATA_DIR / 'summary_dfs',
    'DETAILS_DIR': BASE_DATA_DIR / 'details',
    
    # Parameters
    'MIN_NUC_SIZE': 200,
    'MIN_NUC_SIZE_20x': 100, 
    'USE_GPU': True, 
    'SAVE_INTERMEDIATE': True, # New Flag to control saving
}

# ==================================
# 2. Cell Tracking and Analysis
# ==================================
TRACKING_CONFIG = {
    # Output Directories (if saving)
    'TRACKED_MASK_DIR': BASE_DATA_DIR / OUTPUT_DIRS['TRACKED_MASKS'],
    'TRACK_DF_DIR': BASE_DATA_DIR / OUTPUT_DIRS['TRACK_DF'],
    'PLOT_DIR': BASE_DATA_DIR / OUTPUT_DIRS['PLOTS'],
    
    # Parameters
    'RUN_NAME': RUN_NAME,
    'EXPERIMENT_INFO': EXTERNAL_PATHS['EXPERIMENT_INFO_CSV'],
    'BT_CONFIG_FILE': "/home/nbahou/myimaging/apoDet/scripts/extras/cell_config.json",
    'BT_CONFIG_20X': "/home/myimaging/apoDet/scripts/extras/cell_config_20x.json",
    'BT_CONFIG_20X_5t': '/home/nbahou/myimaging/apoDet/scripts/extras/cell_config_5_20x.json',
    'EPS_TRACK': 70,                                # Tracking radius in pixels
    'EPS_TRACK_20x': 30,
    'TRK_MIN_LEN': 25, 
    'SAVE_INTERMEDIATE': True, # We usually want to save the final track data
}

# ==================================
# 3. Apoptosis Annotation Matching
# ==================================
APO_MATCH_CONFIG = {
    # External Input
    'APO_ANNOTATIONS_DIR': EXTERNAL_PATHS['APO_ANNOTATIONS_DIR'], 

    # Output Directories (if saving)
    'CSV_DIR': BASE_DATA_DIR / OUTPUT_DIRS['APO_MATCH_CSV'],
    'PLOT_DIR': BASE_DATA_DIR / OUTPUT_DIRS['PLOTS'],
    
    # Parameters
    'RUN_NAME': RUN_NAME,
}

# ==================================
# 4. Apoptosis Window Cropping
# ==================================
APO_CROP_CONFIG = {
    # Output Directories (Final Products)
    'WINDOWS_DIR': BASE_DATA_DIR / OUTPUT_DIRS['WINDOW_CROPS_BASE'],
    'WINDOWS_DIR_20X': BASE_DATA_DIR / OUTPUT_DIRS['WINDOW_CROPS_20X_BASE'],
    'CROPS_DIR': BASE_DATA_DIR / OUTPUT_DIRS['CROPS_DIR'],
    'BAD_CROPS': BASE_DATA_DIR / OUTPUT_DIRS['REJECTED_CROPS'],
    'FEATURES_DIR': BASE_DATA_DIR / OUTPUT_DIRS['FILTER_FEATURES'],
    'APO_CHECK_ARRAY_DIR': BASE_DATA_DIR / OUTPUT_DIRS['APO_CHECK_ARRAYS'],
    'PLOT_DIR': BASE_DATA_DIR / OUTPUT_DIRS['PLOTS'],
    
    # Parameters
    'RUN_NAME': RUN_NAME,
    'MAX_TRACKING_DURATION': 20,
    'FRAME_INTERVAL': 5,
    'WINDOW_SIZE': 48,
    'WINDOW_SIZE_20X': 32,
    'ECCENTRICITY_THR': 0.35,
    'SOLIDITY_THR': 0.925,
    'CROP_STD_THR': 1700,
    'CROP_MEAN_INT_THR': 6500,
    'NUM_BLOCKED_FRAMES': 50,

}

# ==================================
# 5. Crop Upsampling (Post-Processing)
# ==================================
UPSAMPLING_CONFIG = {
    'PARENT_DIR': BASE_DATA_DIR,
    'TARGET_SIZE': (128, 128),

    # INPUT: Dynamically generated from the Cropping output base
    'INPUT_WINDOW_DIR_BASE': OUTPUT_DIRS['WINDOW_CROPS_20X_BASE'], 
    # OUTPUT: Dynamically generated from the new output base name
    'OUTPUT_WINDOW_DIR_BASE': OUTPUT_DIRS['WINDOW_CROPS_UPSAMPLED'],

    'CLASS_MAPPINGS': [
        # Define mappings using only the final subdirectory name
        {'NAME': 'apoptotic', 'SUBDIR': 'apo'},
        {'NAME': 'non_apoptotic', 'SUBDIR': 'no_apo'},
        # Add 'random' here if needed later: {'NAME': 'random', 'SUBDIR': 'random'},
    ]
}