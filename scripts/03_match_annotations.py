"""
Apoptosis Annotation Matching and Evaluation

This module matches manually annotated apoptotic events with automatically detected
cell nuclei from a segmentation pipeline. It evaluates the accuracy of automated
detection against expert annotations and generates visualizations of matching metrics.

The workflow includes:
1. Loading manual annotations of apoptotic events
2. Loading segmentation masks and tracking data
3. Matching manual annotations with automatically detected nuclei
4. Calculating matching metrics (success rate, distances)
5. Generating histograms of matching distances

The module compares two matching approaches and provides visualization of the distance
distributions between manual annotations and automated detections.

Requirements:
- numpy, pandas
- matplotlib
- preprocessing module with custom functions (get_image_paths, match_annotations)

Inputs:
- Manual apoptosis annotations (.csv)
- Segmentation masks (.npz)
- Tracked masks with object IDs (.npz)
- Segmentation details (.pkl)

Outputs:
- Matched annotation files (.csv): Manual annotations with corresponding automated detections
- Distance histograms (.png): Visualization of matching accuracy

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
import pandas as pd
# Visualization
import matplotlib.pyplot as plt

from preprocessing import (
    get_image_paths, 
    match_annotations, 
    APO_MATCH_CONFIG,
    get_experiment_info
)


# -- Variables --
# Input Paths
IMG_DIR = APO_MATCH_CONFIG['IMG_DIR']   # For filenames
APO_DIR = APO_MATCH_CONFIG['APO_DIR']
DETAILS_DIR = APO_MATCH_CONFIG['DETAILS_DIR']
MASK_DIR = APO_MATCH_CONFIG['MASK_DIR'] # Stardist label predictions    
TRACKED_MASK_DIR = APO_MATCH_CONFIG['TRACKED_MASK_DIR']
EXPERIMENT_INFO = APO_MATCH_CONFIG['EXPERIMENT_INFO']
# Output Paths
CSV_DIR = APO_MATCH_CONFIG['CSV_DIR']   # File with manual + stardist centroids
# Define Plot directory
PLOT_DIR = APO_MATCH_CONFIG['PLOT_DIR']
RUN_NAME = APO_MATCH_CONFIG['RUN_NAME']


# Logger Set Up (Module-Level)
logger = logging.getLogger(__name__)
# Get the current timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Define log directory and ensure it exists
LOG_DIR = APO_MATCH_CONFIG['LOG_DIR']   # Folder for logs
os.makedirs(LOG_DIR, exist_ok=True)  # Create directory if it doesn't exist

LOG_FILENAME = f"match_annos_{timestamp}.log"
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

# Only forward Warnings/Errors/Critical from btrack
logging.getLogger('btrack').setLevel(logging.WARNING)


# Load image file names in specified directory
logger.info("Starting Image Processing")
image_paths = get_image_paths(os.path.join(IMG_DIR))
filenames = [os.path.splitext(os.path.basename(path))[0]
             for path in image_paths]
logger.info(f"Detected {len(filenames)} files in specified directories.")

# Load Experiment info
try:
    experiments_list = pd.read_csv(EXPERIMENT_INFO, header=0)
except FileNotFoundError as e:
    logger.critical("Critical error: experiment list file not found: "
                    f"{EXPERIMENT_INFO}. Aborting script.")
    sys.exit(1)

# Create directories for saving if they do not exist
output_dirs = [CSV_DIR, PLOT_DIR]
for path in output_dirs:
    os.makedirs(path, exist_ok=True)


# Lists and counters for evaluation of the matching and cropping process
# Initialize list to collect distances between stardist and manual annotations for evaluation
dist_paolo_stardist = []
dist_alt_matching = []
# Initialize counter for evaluation of num matches/mismatches
num_matches_total = 0
num_mismatches_total = 0
# Loop over all files in target directory (predict labels, track and crop windows for each)
logger.info("Starting to process files.")
for path, filename in zip(image_paths, filenames):
    logger.info(f"Processing {filename}")

    # -- Data Loading for current file --
    logger.info("\tMatching manual and stardist annotations")

    # Load Manual Apoptosis annots from csv
    apo_file = os.path.join(APO_DIR, f'{filename}.csv')
    apo_annotations = pd.read_csv(apo_file,
                                  header=None,
                                  names=['filename', 'x', 'y', 't'],
                                  on_bad_lines='skip')
    apo_annotations = apo_annotations.dropna()  # remove rows with NA

    # Determine acq_freq of current file
    exp_info = get_experiment_info(filename, experiments_list)
    # Set multiplier to account for varying acq_freqs
    if not exp_info['found']:
        multiplier = 1
    else:
        dt_acq = int(exp_info['acquisition_freq'])
        dt_apo_annots = int(exp_info['apo_annotation_freq'])
        multiplier = dt_apo_annots//dt_acq
    
    # Load details file from stardist (with centroids, Obj_ID, etc.)
    details_path = os.path.join(DETAILS_DIR, f'{filename}.pkl')
    with open(details_path, 'rb') as f:
        details = pickle.load(f)

    # Load mask file with track_IDs
    mask_path = os.path.join(TRACKED_MASK_DIR, f'{filename}.npz')
    loaded_data = np.load(mask_path)  # Load the .npz file
    tracked_masks = loaded_data['gt']

    # Load mask file with object_IDs
    mask_path = os.path.join(MASK_DIR, f'{filename}.npz')
    loaded_data = np.load(mask_path)  # Load the .npz file
    gt_filtered = loaded_data['gt']

    # -- Matching --
    # Matching with function from preprocessing module
    apo_annotations, metrics = match_annotations(apo_annotations,
                                                 details,
                                                 tracked_masks,
                                                 gt_filtered,
                                                 multiplier)
    # Extract num matches/mismatches
    num_matches = metrics['num_matches']
    num_mismatches = metrics['num_mismatches']
    # Log num matches/mismatches
    logger.info(f"\t\tFound {num_matches} matches and {num_mismatches} mismatches.")
    logger.info(f"\t\t{(num_matches*100)/(num_matches+num_mismatches)}% Success Rate")
    # Update counter for total matches/mismatches
    num_matches_total += num_matches
    num_mismatches_total += num_mismatches

    # Extract distances achieved with custom and L2 matching for eval
    dist_paolo_stardist.append(metrics['dist_paolo_stardist'])
    dist_alt_matching.append(metrics['dist_alt_matching'])

    # Apply previously defined multiplier accounting for varying acq_freqs
    apo_annotations['correct_t'] = apo_annotations['t']*multiplier

    # Save output as CSV with added stardist centroids
    apo_annotations.to_csv(os.path.join(CSV_DIR, f'{filename}.csv'),
                           index=False)
    apo_path = os.path.join(CSV_DIR, f'{filename}.csv')
    logger.info(f"\t\tApo-Annotations with new centroids saved at: {apo_path}")

logger.info("Matching finished")
logger.info(f"\tTotal matches found: {num_matches_total}")
logger.info(f"\tTotal mismatches found: {num_mismatches_total}")


# -- Plotting --
output_dir = os.path.join(PLOT_DIR, RUN_NAME)
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

# Flatten the distance list of lists for plotting
dist_paolo_stardist_flat = np.concatenate(dist_paolo_stardist).tolist()
dist_alt_matching_flat = np.concatenate(dist_alt_matching).tolist()

# Create histogram with matching distances for custom matching
plt.figure(figsize=(8, 6))
plt.hist(dist_paolo_stardist_flat,
         bins=25,
         range=(0, 50),
         color='blue',
         edgecolor='black',
         alpha=0.7)
plt.xlim(0, 50)
# Labels and title
plt.xlabel('Distance (L2-norm, Pixels)')
plt.ylabel('Frequency')
plt.title('Distances between manual and Stardist centroids')
# Save the plot
plot_filename = os.path.join(output_dir, "distances_histogram.png")
plt.savefig(plot_filename)
plt.close()


# Plot histogram for comparison between custom and L2 matching
plt.figure(figsize=(10, 5))
plt.hist(dist_paolo_stardist_flat,
         bins=30,
         alpha=0.5,
         label="Original Matching",
         color='blue',
         range=(0, 60))
plt.hist(dist_alt_matching_flat,
         bins=30,
         alpha=0.5,
         label="L2 Norm Matching",
         color='red',
         range=(0, 60))
plt.xlabel("Distance")
plt.ylabel("Frequency")
plt.title("Comparison of Matching Approaches")
plt.legend()
# Save the plot
plot_filename = os.path.join(output_dir, "matching_comparison_histogram.png")
plt.savefig(plot_filename)
plt.close()
