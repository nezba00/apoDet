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

from preprocessing import (load_image_stack,
                           get_image_paths,
                           crop_window,
                           check_temporal_compatibility)


# from typing import Tuple, Union



## Variables
## Directory Paths
# Input
IMG_DIR = '/mnt/imaging.data/PertzLab/apoDetection/TIFFs'
APO_DIR = '/mnt/imaging.data/PertzLab/apoDetection/ApoptosisAnnotation'
EXPERIMENT_INFO = '/mnt/imaging.data/PertzLab/apoDetection/List of the experiments.csv'
CSV_DIR = '../data/apo_match_csv_test'    # File with manual and stardist centroids
TRACKED_MASK_DIR = '../data/tracked_masks'
TRACK_DF_DIR = '../data/track_dfs'


# Output
MASK_DIR = '../data/apo_masks'    # Stardist label predictions
DF_DIR = '../data/summary_dfs'
CROPS_DIR = '../data/apo_crops_test'    # Directory with .tif files for QC
WINDOWS_DIR = '/home/nbahou/myimaging/apoDet/data/windows_test'    # Directory with crops for scDINO
RANDOM_DIR = os.path.join(WINDOWS_DIR, 'random')
#CLASS_DCT_PATH = './extras/class_dicts'
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

log_filename = f"crop_windows_{timestamp}.log"
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

# Create directories for saving if they do not exist
output_dirs = [CSV_DIR, MASK_DIR, DF_DIR, CROPS_DIR, WINDOWS_DIR]
for path in output_dirs:
    os.makedirs(path, exist_ok=True)
    

# Frame rate and movie length for crops
try:
    experiments_list = pd.read_csv(EXPERIMENT_INFO, header=0)
except FileNotFoundError as e:
    logger.critical(f"Critical error: experiment list file not found: {EXPERIMENT_INFO}. Aborting script.")
    sys.exit(1)
num_timepoints = MAX_TRACKING_DURATION // FRAME_INTERVAL    # eg 20mins/5min = 4 images. 

## Lists and counters for evaluation of the matching and cropping process
# initalize a list to investigate track lengths after apoptosis
survival_times = []


# Loop over all files in target directory (predict labels, track and crop windows for each)
logger.info("Starting to process files.")
for path, filename in zip(image_paths, filenames):
    logger.info(f"Processing {filename}")

    try:
        merge_df_path = os.path.join(TRACK_DF_DIR, f"{filename}.csv")
        merged_df = pd.read_csv(merge_df_path)
        merged_df_long = merged_df[merged_df.groupby("track_id")["track_id"].transform('size') >= TRK_MIN_LEN].copy()
    except:
        logger.warning(f"\tSkipping {filename} due to: {str(e)}. DF with track_ids and centroids not found.")
        continue 
    
    try:
        csv_path = os.path.join(CSV_DIR, f'{filename}.csv')
        apo_annotations = pd.read_csv(csv_path)
    except Exception as e:
        logger.warning(f"\tSkipping {filename} due to: {str(e)}. Annotations of apoptotic cells not found.")
        continue 

    # Load mask with track_ids
    mask_path = os.path.join(TRACKED_MASK_DIR, f'{filename}.npz')
    loaded_data = np.load(mask_path)
    tracked_masks = loaded_data['gt']


    is_valid, result = check_temporal_compatibility(filename, experiments_list, FRAME_INTERVAL)
    print(f"Valid Stack = {is_valid}: acq_freq = {result}")

    # Skip current file if it is not compatible temporally
    if not is_valid:
        logger.warning(f"\t{result}")
        continue

    # Vars for window cropping
    acquisition_freq = result
    step = FRAME_INTERVAL // acquisition_freq    # e.g. 5 // 1 = 1 image every 5 frames
    num_frames = MAX_TRACKING_DURATION // acquisition_freq
    target_size = WINDOW_SIZE if WINDOW_SIZE%2 != 0 else WINDOW_SIZE + 1    # Adds + 1 if even to have the annotated pixel centered in the window

    logger.debug(f"\tTime Info:")
    logger.debug(f"\t\t File Interval: {acquisition_freq} min/image")
    logger.debug(f"\t\tTarget Interval: {FRAME_INTERVAL} min/img")
    logger.debug(f"\t\tStep Size: {step}")
    logger.debug(f"\t\t Imgs in Crops: {num_timepoints} -> +1 for t0")

    # Create directory for cropped windows
    os.makedirs(os.path.join(CROPS_DIR, filename), exist_ok=True)
    os.makedirs(os.path.join(WINDOWS_DIR, 'apo'), exist_ok=True)
    os.makedirs(os.path.join(CROPS_DIR, f'no_apo_{filename}'), exist_ok=True)
    os.makedirs(os.path.join(WINDOWS_DIR, 'no_apo'), exist_ok=True)
    os.makedirs(os.path.join(CROPS_DIR, f'random_{filename}'), exist_ok=True)
    os.makedirs(os.path.join(WINDOWS_DIR, 'random'), exist_ok=True)


    # Load images to extract windows
    imgs = load_image_stack(os.path.join(IMG_DIR, f'{filename}.tif'))
    
    # Counter so we can sample same number of random tracks as apoptotic
    num_apo_crops = 0 

    # Initialize column apoptotic with zeros
    merged_df_long['apoptotic'] = 0
    
    apo_track_ids = pd.DataFrame(columns=['track_id', 'apo_start_t'])

    logger.info("\tStarting cropping for apo cells")
    for i, row in tqdm(apo_annotations.iterrows(), total=len(apo_annotations), desc="Processing Annotations"):
        current_track_id = row.loc['matching_track']
        current_t = row.loc['t'] + row.loc['delta_ts']

        apo_track_ids.loc[i, 'track_id'] = current_track_id
        apo_track_ids.loc[i, 'apo_start_t'] = current_t

        #current_track_id = current_track_id.squeeze()
        if not np.isscalar(current_track_id):
            current_track_id = current_track_id.iloc[0]

        
        single_cell_df = merged_df_long.loc[(merged_df_long['track_id'] == current_track_id) & (merged_df_long['t'] >= current_t)]
        if len(single_cell_df) < num_frames + 1:
            logger.warning(f"\t\tSkipping current track: {current_track_id}. Reason: Track lost too quickly after anno. len = {len(single_cell_df)}")
            continue
        
        # Count track length after manual apoptosis annotation (for histogram later)
        num_entries = single_cell_df.shape[0]
        survival_times.append(num_entries)

        # merged_df_long.loc[single_cell_df.index, 'apoptotic'] = 1
        merged_df_long.loc[(merged_df_long['track_id'] == current_track_id) & (merged_df_long['t'] >= current_t), 'apoptotic'] = 1
        single_cell_df = single_cell_df.loc[single_cell_df['t'] < current_t + num_frames + step]
        
        
        windows = []
        for _, sc_row in single_cell_df.iterrows():
            window = crop_window(imgs[int(sc_row['t'])], int(sc_row['x']), int(sc_row['y']), WINDOW_SIZE)
            if window.shape == (target_size, target_size):    ### This is a bit dangerous! can mess up time if we simply leave a window out
                windows.append(window)
            else:
                windows.append(None)


        chosen_offset = None
        for offset in range(step):
            sub_windows = windows[offset::step]
            if all(x is not None for x in sub_windows):  
                chosen_offset = offset
                break  # Stop as soon as a valid sequence is found

        if chosen_offset is None:
            logger.warning("\t\tNo valid start point for a sequence with correct intervals found.")
        else:
            sub_windows = windows[chosen_offset::step]
            sub_windows = np.asarray(sub_windows)
            if len(sub_windows) == (MAX_TRACKING_DURATION // step) + 1:
                tiff.imwrite(os.path.join(CROPS_DIR, filename, f'trackID_{current_track_id}.tif'), sub_windows)
                tiff.imwrite(os.path.join(WINDOWS_DIR, 'apo', f'apo_{filename}_{i}.tif'), sub_windows)
                num_apo_crops += 1
            else:
                logger.warning(f'\t\tAt least one of the images has the wrong size after cropping and could not be used. Length = {len(windows)}. Pos = {sc_row["x"]}, {sc_row["y"]}.')
    logger.info(f"\t\tValid crops of apo cells found for {num_apo_crops}/{len(apo_annotations)}")

    num_random_tracks = num_apo_crops
    
    logger.info('\tStarting cropping for non-apo cells.')
    # Get crops of healthy cells
    long_no_apo_df = merged_df_long[merged_df_long['apoptotic'] == 0]
    unique_track_ids = np.unique(long_no_apo_df['track_id'])

    num_healthy_crops = 0
    rejected_windows = 0

    for i, track_id in tqdm(enumerate(unique_track_ids), total=len(unique_track_ids), desc="Cropping non-apo Windows"):
        single_cell_df = long_no_apo_df.loc[long_no_apo_df['track_id'] == track_id]
        start_t = min(single_cell_df['t'])
        # same length as for apoptotic cells
        single_cell_df = single_cell_df.loc[single_cell_df['t'] <= start_t + num_frames]
        windows = []
        for _, sc_row in single_cell_df.iterrows():
            window = crop_window(imgs[int(sc_row['t'])], int(sc_row['x']), int(sc_row['y']), WINDOW_SIZE)
            if window.shape == (target_size, target_size):    ### This is a bit dangerous! can mess up time if we simply leave a window out
                windows.append(window)

        if len(windows) == (MAX_TRACKING_DURATION) + 1:
            windows = np.asarray(windows)
            tiff.imwrite(os.path.join(CROPS_DIR, f'no_apo_{filename}', f'trackID_{track_id}.tif'), windows[::step])
            tiff.imwrite(os.path.join(WINDOWS_DIR, 'no_apo', f'no_apo_{filename}_{i}.tif'), windows[::step])
            num_healthy_crops += 1
        else:
            rejected_windows += 1

            logger.debug(f'\t\tAt least one of the images has the wrong size. Length = {len(windows)}. Pos = {sc_row["x"]}, {sc_row["y"]}.')
    logger.info(f"\t\tFound {num_healthy_crops} valid crops of healthy cells.")
    if rejected_windows > 0:
        logger.warning(f"\t\t{rejected_windows} windows were rejected due to incorrect size.")
    
    # Gather random crops 
    ### I think this could be problematic, oftentimes is full of cells so not really background or negative
    logger.info(f'\tStarting cropping for {num_random_tracks} random locations.')

    # Assuming imgs is a list/array of your image frames,
    # and each image has dimensions: img_height x img_width
    img_height, img_width = imgs[0].shape

    iter_count = 0
    crop_count = 0
    while (crop_count <= num_random_tracks) and (iter_count <= 100):    # Iter count only to stop runaway code
    # for track in range(num_random_tracks):
        # Randomly choose a start time such that the track fits in your image sequence
        start_t = np.random.randint(0, len(imgs) - num_frames)
        
        # Randomly generate valid (x, y) coordinates for the crop center
        random_x = np.random.randint(WINDOW_SIZE//2, img_width - WINDOW_SIZE//2)
        random_y = np.random.randint(WINDOW_SIZE//2, img_height - WINDOW_SIZE//2)
        
        windows = []
        # Extract a window for each frame in the track duration
        for t in range(start_t, start_t + num_frames + 1, step):
            window = crop_window(imgs[t], random_x, random_y, WINDOW_SIZE)
            track_id_mask_crop = crop_window(tracked_masks[t], random_x, random_y, WINDOW_SIZE)
            present_track_ids = track_id_mask_crop.flatten().tolist()
            

            ### Continue here
            current_apo_track_ids = apo_track_ids.loc[apo_track_ids['apo_start_t'] <= t]
            current_apo_ids = set(current_apo_track_ids['track_id'])

            
            if not any(pixel in current_apo_ids for pixel in present_track_ids) and (window.shape == (target_size, target_size)):
                windows.append(window)
            else:
                logger.warning("\t\tCurrent window contains an apoptotic cell or the shape is not matching the target size.")

    
        if len(windows) == (MAX_TRACKING_DURATION//FRAME_INTERVAL) + 1:
            windows = np.asarray(windows)
            tiff.imwrite(os.path.join(CROPS_DIR, f'random_{filename}', f'ID_{crop_count}.tif'), windows)
            tiff.imwrite(os.path.join(WINDOWS_DIR, 'random', f'random_{filename}_{crop_count}.tif'), windows)
            crop_count += 1
            iter_count += 1
        else:
            logger.warning(f'\t\tUnable to collect the desired number of images. Length windows = {len(windows)}')
            iter_count += 1

    logger.info(f"Finished processing {filename}: apo ({num_apo_crops}), healthy ({num_healthy_crops}), random ({num_random_tracks})")


### Plotting



# Define the output directory
output_dir = os.path.join(PLOT_DIR, RUN_NAME)
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

# Create the plot
plt.figure(figsize=(8, 6))
plt.hist(survival_times, bins=range(min(survival_times), max(survival_times)+2), edgecolor='black')
plt.xlabel("Survival Time (frames)")
plt.ylabel("Number of Cells")
plt.title("Histogram of Cell Survival Times (Frames)")

# Save the plot to a file
plot_filename = os.path.join(output_dir, "survival_times_histogram.png")
plt.savefig(plot_filename)

plt.close()



plt.figure(figsize=(8, 6))
plt.hist(survival_times, bins=20, range=(0, 200), edgecolor='black')
plt.xlim(0, 200)
plt.xlabel("Survival Time (frames)")
plt.ylabel("Number of Cells")
plt.title("Histogram of Cell Survival Times (Frames)")

# Save the plot to a file
plot_filename = os.path.join(output_dir, "survival_times_histogram_option1.png")
plt.savefig(plot_filename)


# Close the plot to free up memory
plt.close()