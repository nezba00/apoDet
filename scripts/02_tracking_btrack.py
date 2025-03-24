"""
Cell Tracking and Analysis Pipeline

This module performs automated tracking of segmented cell nuclei using the BTrack algorithm.
It processes previously segmented masks, applies tracking to establish temporal relationships
between detected objects, and generates visualization of track statistics.

The workflow includes:
1. Loading segmented masks from a specified directory
2. Applying BTrack algorithm to track nuclei across time frames
3. Merging tracking information with original segmentation data
4. Converting object IDs to track IDs in masks
5. Generating and saving track length histograms

The module requires pre-generated segmentation masks and summary dataframes from the
segmentation pipeline. It uses BTrack for object tracking with configurable parameters
for tracking radius and minimum track length.

Requirements:
- numpy, pandas
- matplotlib
- btrack (tracking algorithm)
- preprocessing module with custom functions (get_image_paths,
                                              run_tracking,
                                              convert_obj_to_track_ids)

Outputs:
- Tracked masks (.npz): Segmentation masks with track IDs
- Track dataframes (.csv): Combined segmentation and tracking information
- Track length histograms (.png): Distribution of track durations

"""

# Imports
# Standard library imports
import logging
import os
import sys
from datetime import datetime


# Third-party imports
# Data handling
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt

# Utilities
from preprocessing import (convert_obj_to_track_ids, get_image_paths,
                           run_tracking)


# Variables
# Directory Paths
# Input
IMG_DIR = '/mnt/imaging.data/PertzLab/apoDetection/TIFFs'
MASK_DIR = '../data/apo_masks'    # Stardist label predictions
DF_DIR = '../data/summary_dfs'

# Output
TRACKED_MASK_DIR = '../data/tracked_masks'
TRACK_DF_DIR = '../data/track_dfs'

# Define plot directory
PLOT_DIR = "/home/nbahou/myimaging/apoDet/data/plots"
RUN_NAME = "test"

# Tracking Parameters
BT_CONFIG_FILE = "extras/cell_config.json"  # Path to btrack config file
EPS_TRACK = 70         # Tracking radius [px]
TRK_MIN_LEN = 25       # Minimum track length [frames]


# Logger Set Up
# logging.shutdown()    # For jupyter notebooks
logger = logging.getLogger(__name__)
# if logger.hasHandlers():
#    logger.handlers.clear()
# Get the current timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Define log directory and ensure it exists
LOG_DIR = "./logs"  # Folder for logs
os.makedirs(LOG_DIR, exist_ok=True)  # Create directory if it doesn't exist

LOG_FILENAME = f"tracking_Btrack_{timestamp}.log"
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


# Load image paths in specified directory
logger.info("Starting Image Processing")
image_paths = get_image_paths(os.path.join(IMG_DIR))
filenames = [os.path.splitext(os.path.basename(path))[0]
             for path in image_paths[:2]]    # TODO remove :2 here, was only for testing
logger.info(f"Detected {len(filenames)} files in specified directories.")
# print(filenames)

# Create directories for saving if they do not exist
output_dirs = [MASK_DIR, DF_DIR, TRACK_DF_DIR, TRACKED_MASK_DIR]
for path in output_dirs:
    os.makedirs(path, exist_ok=True)

# Loop over all files in target directory (predict labels, track and crop windows for each)
logger.info("Starting to process files.")
for path, filename in zip(image_paths, filenames):
    # Define and load labels
    mask_path = os.path.join(MASK_DIR, f'{filename}.npz')
    with np.load(mask_path) as data:
        gt_filtered = data['gt']  # Access the saved array

    df_path = os.path.join(DF_DIR, f'{filename}_pd_df.csv')
    strdst_df = pd.read_csv(df_path, header=0)

    # Run tracking with Btrack
    _, fovY, fovX = gt_filtered.shape
    dfBTracks = run_tracking(gt_filtered, fovX, fovY, BT_CONFIG_FILE, EPS_TRACK)

    logger.info("\tMerging information from Btrack and stardist.")
    merged_df = strdst_df.merge(dfBTracks.drop(columns=["x", "y", "area"]),
                                on=["obj_id", "t"],
                                how="left")
    # Enable next lines if you want tracks which are longer than TRK_MIN_LEN
    # track_sizes = merged_df.groupby("track_id")["track_id"].transform('size')
    # merged_df = merged_df[track_sizes >= TRK_MIN_LEN].copy()

    logger.info("\t\tComplete.")

    # Save merged DataFrame to a CSV file
    merge_df_path = os.path.join(TRACK_DF_DIR, f"{filename}.csv")
    merged_df.to_csv(merge_df_path, index=False)
    logger.info(f"\tSaved merged dfs at: {merge_df_path}")

    # Create a mask with btrack track_ids instead of stardists obj_ids
    logger.info("\tConverting Obj_IDs to Track_IDs in stardist masks.")
    tracked_masks = convert_obj_to_track_ids(gt_filtered, merged_df)

    # Save masks
    mask_path = os.path.join(TRACKED_MASK_DIR, f'{filename}.npz')
    np.savez_compressed(mask_path, gt=tracked_masks)
    logger.info(f"\t\tMask saved at: {mask_path}")


output_dir = os.path.join(PLOT_DIR, RUN_NAME)
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

# Count occurrences of each track_id to determine track lengths
track_lengths = merged_df["track_id"].value_counts()

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(track_lengths, bins=20, edgecolor="black", alpha=0.7)

# Labels and title
plt.xlabel("Track Length")
plt.ylabel("Frequency")
plt.title("Histogram of Track Lengths")

# Save the plot
plot_filename = os.path.join(output_dir, "track_lengths_histogram.png")
plt.savefig(plot_filename)

# Close the plot to free up memory
plt.close()

track_sizes = merged_df.groupby("track_id")["track_id"].transform('size')
merged_df_long = merged_df[track_sizes >= TRK_MIN_LEN].copy()

# Count occurrences of each track_id to determine track lengths
track_lengths = merged_df_long["track_id"].value_counts()

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(track_lengths, bins=20, edgecolor="black", alpha=0.7)

# Labels and title
plt.xlabel("Track Length")
plt.ylabel("Frequency")
plt.title("Histogram of Track Lengths")

# Save the plot
plot_filename = os.path.join(output_dir, "track_lengths_histogram_long.png")
plt.savefig(plot_filename)

# Optionally, display the plot
plt.show()

# Close the plot to free up memory
plt.close()
