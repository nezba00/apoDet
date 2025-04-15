"""
Nuclear Segmentation Using StarDist

This module performs segmentation of cell nuclei in fluorescence microscopy
images using StarDist algorithm. It processes images from a specified directory,
applies segmentation, filters results based on size criteria, and saves masks and
metadata for downstream analysis.

The workflow includes:
1. Loading images from a specified directory
2. Applying StarDist segmentation to detect nuclei
3. Filtering objects below a minimum size threshold
4. Saving segmentation masks and metadata (summary dataframes and detailed results)

The module uses a pre-trained StarDist model ("2D_versatile_fluo") and can leverage
GPU acceleration when available.

Requirements:
- stardist
- numpy
- gputools (if GPU processing is enabled)
- csbdeep (for GPU memory management)
- preprocessing module with custom functions

Outputs:
- Binary masks (.npz): Segmented nuclei masks
- Summary dataframes (.csv): Object ID, time point, and position data
- Detailed results (.pkl): Complete segmentation details (centroids and more)

"""

# Imports
# Standard library imports
import logging
import os
import pickle
import sys
from datetime import datetime

# Third-party imports
# Data handling
import numpy as np
# Deep learning and segmentation
from stardist.models import StarDist2D

from preprocessing import (filter_segmentation, get_image_paths,
                           run_segmentation, SEGMENTATION_CONFIG)

# Variables
# Directory Paths
# Input
IMG_DIR = SEGMENTATION_CONFIG['IMG_DIR']
# Output
MASK_DIR = SEGMENTATION_CONFIG['MASK_DIR']  # Stardist label predictions
MASK_DIR_NO_FILT = SEGMENTATION_CONFIG['MASK_DIR_NO_FILT']
DF_DIR = SEGMENTATION_CONFIG['DF_DIR']
DETAILS_DIR = SEGMENTATION_CONFIG['DETAILS_DIR']
LOG_DIR = SEGMENTATION_CONFIG['LOG_DIR']  # Folder for logs


# Processing Configuration
SAVE_DATA = SEGMENTATION_CONFIG['SAVE_DATA']
USE_GPU = SEGMENTATION_CONFIG['USE_GPU']

MIN_NUC_SIZE = SEGMENTATION_CONFIG['MIN_NUC_SIZE']  # Removes objects < x


# Logger Set Up
logger = logging.getLogger(__name__)
# Get the current timestamp
# Define log directory and ensure it exists
os.makedirs(LOG_DIR, exist_ok=True)  # Create directory if it doesn't exist
# Define Logname
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_FILENAME = f"strdst_segment_{timestamp}.log"
log_path = os.path.join(LOG_DIR, LOG_FILENAME)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(sys.stdout)  # Outputs to console too
    ],
    force=True
)

# Create a logger instance
logger = logging.getLogger(__name__)


# Load image paths in specified directory
logger.info("Starting Image Processing")
image_paths = get_image_paths(os.path.join(IMG_DIR))
filenames = [os.path.splitext(os.path.basename(path))[0]
             for path in image_paths]
num_files = len(filenames)
logger.info(f"Detected {len(filenames)} files in specified directories.")

# Create directories for saving if they do not exist
output_dirs = [MASK_DIR, MASK_DIR_NO_FILT, DF_DIR, DETAILS_DIR]
for path in output_dirs:
    os.makedirs(path, exist_ok=True)

# Set up GT_mask prediction with stardist
if USE_GPU:
    import gputools
    from csbdeep.utils.tf import limit_gpu_memory
    limit_gpu_memory(None, allow_growth=True)
model = StarDist2D.from_pretrained("2D_versatile_fluo")    # Load standard model to create GT
axis_norm = (0, 1)    # for normalization


# Loop over all files in target directory (predict labels, track and crop windows for each)
logger.info("Starting Segmentation.")
for path, filename in zip(image_paths, filenames):
    logger.info(f"Processing {filename}")

    # Stardist nuclei segmentation
    # Load and normalize image stack
    logger.info("\tStarting with Stardist segmentation.")
    gt, details = run_segmentation(path, model, axis_norm)
    logger.info("\t\tSegmentation done.")

    # Remove small objects and create DF with segmentation info (object_id, t, x, y)
    gt_filtered, summary_df = filter_segmentation(gt, details, MIN_NUC_SIZE)

    # Save data
    if SAVE_DATA:
        # Save masks
        mask_path = os.path.join(MASK_DIR, f'{filename}.npz')
        no_filt_path = os.path.join(MASK_DIR_NO_FILT, f'{filename}.npz')
        np.savez_compressed(no_filt_path, gt=gt)
        np.savez_compressed(mask_path, gt=gt_filtered)
        logger.info(f"\t\tMask saved at: {mask_path}")
        # Save summary df (obj_id, t, x, y for every detected object)
        summary_df.to_csv(
            os.path.join(DF_DIR, f'{filename}_pd_df.csv'),
            index=False
        )
        logger.info(f"\t\tSummary-Df saved at: {os.path.join(DF_DIR, f'{filename}_pd_df.csv')}")

        details_path = os.path.join(DETAILS_DIR, f'{filename}.pkl')
        with open(details_path, 'wb') as f:
            pickle.dump(details, f)
