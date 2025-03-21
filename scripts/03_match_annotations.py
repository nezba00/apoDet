### Imports
# Standard library imports
import json
import os
import shutil
import logging
import sys

# Third-party imports
# Data handling
import numpy as np
import pandas as pd

# Image I/O and processing
import tifffile as tiff
from nd2reader import ND2Reader
from skimage.morphology import remove_small_objects


# Tracking
import btrack
from btrack.constants import BayesianUpdates

# Visualization
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# Utilities
from tqdm import tqdm

from datetime import datetime

import pickle

from preprocessing import (load_image_stack,
                           get_image_paths,
                           match_annotations)



## Variables
## Directory Paths
# Input
IMG_DIR = '/mnt/imaging.data/PertzLab/apoDetection/TIFFs'
APO_DIR = '/mnt/imaging.data/PertzLab/apoDetection/ApoptosisAnnotation'
EXPERIMENT_INFO = '/mnt/imaging.data/PertzLab/apoDetection/List of the experiments.csv'
DETAILS_DIR = '../data/details_test'
TRACKED_MASK_DIR = '../data/tracked_masks'

# Output
MASK_DIR = '../data/apo_masks'    # Stardist label predictions
CSV_DIR = '../data/apo_match_csv_test'    # File with manual and stardist centroids
DF_DIR = '../data/summary_dfs'
CROPS_DIR = '../data/apo_crops_test'    # Directory with .tif files for QC
WINDOWS_DIR = '/home/nbahou/myimaging/apoDet/data/windows_test'    # Directory with crops for scDINO
RANDOM_DIR = os.path.join(WINDOWS_DIR, 'random')
CLASS_DCT_PATH = './extras/class_dicts'
# Define Plot directory
PLOT_DIR = "/home/nbahou/myimaging/apoDet/data/plots"
RUN_NAME = "test"


## Processing Configuration
COMPARE_2D_VERS = True
SAVE_MASKS = True
LOAD_MASKS = True
USE_GPU = True
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

log_filename = f"match_annos_{timestamp}.log"
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

# Only forward Warnings/Errors/Critical from btrack
logging.getLogger('btrack').setLevel(logging.WARNING)


# Load image paths in specified directory
logger.info("Starting Image Processing")
image_paths = get_image_paths(os.path.join(IMG_DIR))
filenames = [os.path.splitext(os.path.basename(path))[0] for path in image_paths[:2]]    ### TODO remove :2 here, was only for testing
logger.info(f"Detected {len(filenames)} files in specified directories.")
#print(filenames)

# TODO Create directories for saving if they do not exist
output_dirs = [CSV_DIR, MASK_DIR, DF_DIR, CROPS_DIR, CLASS_DCT_PATH, WINDOWS_DIR]
for path in output_dirs:
    os.makedirs(path, exist_ok=True)


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
logger.info("Starting to process files.")
for path, filename in zip(image_paths, filenames):
    logger.info(f"Processing {filename}")



    ### Match manual and stardist annotations
    logger.info("\tMatching manual and stardist annotations")
    # Read CSV with manual apoptosis annotations 
    apo_file = os.path.join(APO_DIR, f'{filename}.csv')
    apo_annotations = pd.read_csv(apo_file, header=None, names=['filename', 'x', 'y', 't'])

    # Read details file from stardist (with centroids, etc.)
    details_path = os.path.join(DETAILS_DIR, f'{filename}.pkl')
    with open(details_path, 'rb') as f:
        details = pickle.load(f)

    # Load mask file with track_ids
    mask_path = os.path.join(TRACKED_MASK_DIR, f'{filename}.npz')
    loaded_data = np.load(mask_path)  # Load the .npz file
    tracked_masks = loaded_data['gt']

    # Load stardist segmentation masks
    mask_path = os.path.join(MASK_DIR, f'{filename}.npz')
    loaded_data = np.load(mask_path)  # Load the .npz file
    gt_filtered = loaded_data['gt']      

    apo_annotations, metrics = match_annotations(apo_annotations, details, tracked_masks, gt_filtered)

    num_matches += metrics['num_matches']
    num_mismatches += metrics['num_mismatches']
    dist_paolo_stardist.append(metrics['dist_paolo_stardist'])
    dist_alt_matching.append(metrics['dist_alt_matching'])
    

    # Save again as CSV with added stardist centroids      
    apo_annotations.to_csv(os.path.join(CSV_DIR, f'{filename}.csv'), index=False)
    logger.info(f"\t\tApo-Annotations with stardist centroids saved at: {os.path.join(CSV_DIR, f'{filename}.csv')}")



# Plotting
output_dir = os.path.join(PLOT_DIR, RUN_NAME)
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

# Flatten the distance lists
dist_paolo_stardist_flat = np.concatenate(dist_paolo_stardist).tolist()
dist_alt_matching_flat = np.concatenate(dist_alt_matching).tolist()

# Create histogram
plt.figure(figsize=(8, 6))
plt.hist(dist_paolo_stardist_flat, bins=25, range=(0, 50), color='blue', edgecolor='black', alpha=0.7)
plt.xlim(0, 50)

# Labels and title
plt.xlabel('Distance (L2-norm, Pixels)')
plt.ylabel('Frequency')
plt.title('Distances between manual and Stardist centroids')

# Save the plot
plot_filename = os.path.join(output_dir, "distances_histogram.png")
plt.savefig(plot_filename)


# Close the plot to free up memory
plt.close()


# Plot histograms
plt.figure(figsize=(10, 5))
plt.hist(dist_paolo_stardist_flat, bins=30, alpha=0.5, label="Original Matching", color='blue', range=(0, 60))
plt.hist(dist_alt_matching_flat, bins=30, alpha=0.5, label="L2 Norm Matching", color='red', range=(0, 60))
plt.xlabel("Distance")
plt.ylabel("Frequency")
plt.title("Comparison of Matching Approaches")
plt.legend()

# Save the plot
plot_filename = os.path.join(output_dir, "matching_comparison_histogram.png")
plt.savefig(plot_filename)


# Close the plot to free up memory
plt.close()