### Imports
# Standard library imports
from preprocessing import (get_image_paths, 
                           run_segmentation, 
                           filter_segmentation)

import os
import logging
import sys

# Third-party imports
# Data handling
import numpy as np

# Deep learning and segmentation
from stardist.models import StarDist2D

from datetime import datetime
import pickle



## Variables
## Directory Paths
# Input
IMG_DIR = '/mnt/imaging.data/PertzLab/apoDetection/TIFFs'


# Output
MASK_DIR = '../data/apo_masks_test'    # Stardist label predictions
DF_DIR = '../data/summary_dfs_test'
DETAILS_DIR = '../data/details_test'




## Processing Configuration
SAVE_DATA = True
USE_GPU = True


LOAD_MASKS = True
MIN_NUC_SIZE = 200

## Tracking Parameters


BT_CONFIG_FILE = "extras/cell_config.json"  # Path to btrack config file
EPS_TRACK = 70         # Tracking radius [px]
TRK_MIN_LEN = 25       # Minimum track length [frames]

#
MAX_TRACKING_DURATION = 20    # In minutes
FRAME_INTERVAL = 5    # minutes between images we want

WINDOW_SIZE = 61


## Logger Set Up
#logging.shutdown()    # For jupyter notebooks
logger = logging.getLogger(__name__)
#if logger.hasHandlers():
#    logger.handlers.clear()
# Get the current timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Define log directory and ensure it exists
log_dir = "./logs"  # Folder for logs
os.makedirs(log_dir, exist_ok=True)  # Create directory if it doesn't exist

log_filename = f"strdst_segment_{timestamp}.log"
log_path = os.path.join(log_dir, log_filename)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(sys.stdout)  # Outputs to console too
    ],
    force = True
)

# Create a logger instance
logger = logging.getLogger(__name__)


# Load image paths in specified directory
logger.info("Starting Image Processing")
image_paths = get_image_paths(os.path.join(IMG_DIR))
filenames = [os.path.splitext(os.path.basename(path))[0] for path in image_paths[:2]]    ### TODO remove :2 here, was only for testing
logger.info(f"Detected {len(filenames)} files in specified directories.")
#print(filenames)

# Create directories for saving if they do not exist
output_dirs = [MASK_DIR, DF_DIR, DETAILS_DIR]
for path in output_dirs:
    os.makedirs(path, exist_ok=True)

# Set up GT_mask prediction with stardist
if USE_GPU:
    import gputools
    from csbdeep.utils.tf import limit_gpu_memory
    limit_gpu_memory(None, allow_growth=True)
model = StarDist2D.from_pretrained("2D_versatile_fluo")    # Load standard model to create GT
axis_norm = (0,1)    # for normalization


### TODO All of this not used here, tracking and matching
## Lists and counters for evaluation of the matching and cropping process
# Initialize list to collect distances between stardist and manual annotations for evaluation
dist_paolo_stardist = []
dist_alt_matching = []
# Initialize counter for evaluation of num matches/mismatches
num_matches = 0
num_mismatches = 0
# initalize a list to investigate track lengths after apoptosis
survival_times = []

# Loop over all files in target directory (predict labels, track and crop windows for each)
logger.info("Starting Segmentation.")
for path, filename in zip(image_paths, filenames):
    logger.info(f"Processing {filename}")
    # Skip file if not in experiment info csv
    experiment_num = filename.split('_')[0]

    ### Stardist nuclei segmentation
    # Load and normalize image stack
    logger.info("\tStarting with Stardist segmentation.")
    gt, details = run_segmentation(path, model, axis_norm)
    logger.info("\t\tSegmentation done.")

    # Remove small objects and create DF with segmentation info (object_id, t, x, y)
    gt_filtered, summary_df = filter_segmentation(gt, details, MIN_NUC_SIZE)
    
    # Save summary df as CSV

    
    # Save labels
    if SAVE_DATA:
        # Save masks
        mask_path = os.path.join(MASK_DIR, f'{filename}.npz')
        np.savez_compressed(mask_path, gt=gt_filtered)
        logger.info(f"\t\tMask saved at: {mask_path}")
        # Save summary df (obj_id, t, x, y for every detected object)
        summary_df.to_csv(os.path.join(DF_DIR, f'{filename}_pd_df.csv'), index=False)
        logger.info(f"\t\tSummary-Df saved at: {os.path.join(DF_DIR, f'{filename}_pd_df.csv')}")

        details_path = os.path.join(DETAILS_DIR, f'{filename}.pkl')
        with open(details_path, 'wb') as f:
            pickle.dump(details, f)

        
