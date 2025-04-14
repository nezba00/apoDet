"""
Apoptosis Window Cropping and Dataset Generation

This module extracts time-series image crops centered on detected cells for
machine learning applications. It creates a balanced dataset of apoptotic,
non-apoptotic, and random cell windows with consistent temporal resolution.

The workflow includes:
1. Loading tracking data and matched annotations
2. Extracting time-series windows centered on apoptotic cells
3. Extracting matched windows of non-apoptotic cells
4. Generating random control windows
5. Creating visualization of survival times after apoptosis

The module requires tracked masks, annotation matches, and experiment metadata.
It generates temporally aligned image crops at specified intervals for
downstream machine learning applications.

Parameters:
- Window size: Size of spatial crops in pixels
- Frame interval: Desired temporal resolution between frames
- Max tracking duration: Total tracking time in minutes

Requirements:
- numpy, pandas
- tifffile
- matplotlib
- tqdm
- preprocessing module with custom functions

Outputs:
- Image crops (.tif): Time-series windows of apoptotic, healthy and random cells
- Survival time histograms (.png): Duration cells remain visible after apoptosis

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

# Image I/O and processing
import tifffile as tiff
from skimage import util, measure

# Visualization
import matplotlib.pyplot as plt

# Utilities
from tqdm import tqdm
from preprocessing import (check_temporal_compatibility, crop_window,
                           get_image_paths, load_image_stack,
                           APO_CROP_CONFIG)


# Variables
# Directory Paths
# Input
IMG_DIR = APO_CROP_CONFIG['IMG_DIR']
EXPERIMENT_INFO = APO_CROP_CONFIG['EXPERIMENT_INFO']
CSV_DIR = APO_CROP_CONFIG['CSV_DIR']    # File with manual and stardist centroids
TRACKED_MASK_DIR = APO_CROP_CONFIG['TRACKED_MASK_DIR']
TRACK_DF_DIR = APO_CROP_CONFIG['TRACK_DF_DIR']

# Output
CROPS_DIR = APO_CROP_CONFIG['CROPS_DIR']    # Directory with .tif files for QC
WINDOWS_DIR = APO_CROP_CONFIG['WINDOWS_DIR']    # Directory with crops for scDINO
BAD_CROPS = APO_CROP_CONFIG['BAD_CROPS']
FEATURES_DIR = APO_CROP_CONFIG['FEATURES_DIR']
APO_CHECK_ARRAY_DIR = APO_CROP_CONFIG['APO_CHECK_ARRAY_DIR']

# Plots
PLOT_DIR = APO_CROP_CONFIG['PLOT_DIR']
RUN_NAME = APO_CROP_CONFIG['RUN_NAME']


# Tracking Parameters
TRK_MIN_LEN = APO_CROP_CONFIG['TRK_MIN_LEN']    # Minimum track length [frames]

# Time settings
MAX_TRACKING_DURATION = APO_CROP_CONFIG['MAX_TRACKING_DURATION']    # In minutes
FRAME_INTERVAL = APO_CROP_CONFIG['FRAME_INTERVAL']  # minutes between images

WINDOW_SIZE = APO_CROP_CONFIG['WINDOW_SIZE']

ECCENTRICITY_THR = APO_CROP_CONFIG['ECCENTRICITY_THR']
SOLIDITY_THR = APO_CROP_CONFIG['SOLIDITY_THR']
CROP_STD_THR = APO_CROP_CONFIG['CROP_STD_THR']
CROP_MEAN_INT_THR = APO_CROP_CONFIG['CROP_MEAN_INT_THR']
NUM_BLOCKED_FRAMES = APO_CROP_CONFIG['NUM_BLOCKED_FRAMES']

# Logger Set Up
logger = logging.getLogger(__name__)
# Get the current timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Define log directory and ensure it exists
LOG_DIR = APO_CROP_CONFIG['LOG_DIR']    # Folder for logs  
os.makedirs(LOG_DIR, exist_ok=True)  # Create directory if it doesn't exist

LOG_FILENAME = f"crop_windows_{timestamp}.log"
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
             for path in image_paths[:2]]
logger.info(f"Detected {len(filenames)} files in specified directories.")

# Create directories for saving if they do not exist
output_dirs = [CROPS_DIR, WINDOWS_DIR, BAD_CROPS, FEATURES_DIR, APO_CHECK_ARRAY_DIR]
for path in output_dirs:
    os.makedirs(path, exist_ok=True)

os.makedirs(os.path.join(FEATURES_DIR, 'raw_images'), exist_ok=True)
os.makedirs(os.path.join(FEATURES_DIR, 'masks'), exist_ok=True)

# Frame rate and movie length for crops
try:
    experiments_list = pd.read_csv(EXPERIMENT_INFO, header=0)
except FileNotFoundError as e:
    logger.critical("Critical error: experiment list file not found: "
                    f"{EXPERIMENT_INFO}. Aborting script.")
    sys.exit(1)
num_timepoints = MAX_TRACKING_DURATION // FRAME_INTERVAL    # eg 20mins/5min = 4 images.

# Lists and counters for evaluation of the matching and cropping process
# initalize a list to investigate track lengths after apoptosis
survival_times = []
all_features = []

# Loop over all files in target directory (predict labels, track and crop windows for each)
logger.info("Starting to process files.")
for path, filename in zip(image_paths, filenames):
    logger.info(f"Processing {filename}")

    try:
        merge_df_path = os.path.join(TRACK_DF_DIR, f"{filename}.csv")
        merged_df = pd.read_csv(merge_df_path)
        track_sizes= merged_df.groupby("track_id")["track_id"].transform('size')
        merged_df_long = merged_df[track_sizes >= TRK_MIN_LEN].copy()
    except Exception as e:
        logger.warning(f"\tSkipping {filename} due to: {str(e)}. "
                       "DF with track_ids and centroids not found.")
        continue

    try:
        csv_path = os.path.join(CSV_DIR, f'{filename}.csv')
        apo_annotations = pd.read_csv(csv_path)
    except Exception as e:
        logger.warning(f"\tSkipping {filename} due to: {str(e)}. "
                       "Annotations of apoptotic cells not found.")
        continue

    # Load mask with track_ids
    mask_path = os.path.join(TRACKED_MASK_DIR, f'{filename}.npz')
    loaded_data = np.load(mask_path)
    tracked_masks = loaded_data['gt']

    is_valid, result = check_temporal_compatibility(filename,
                                                    experiments_list,
                                                    FRAME_INTERVAL)
    # print(f"Valid Stack = {is_valid}: acq_freq = {result}")

    # Skip current file if it is not compatible temporally
    if not is_valid:
        logger.warning(f"\t{result}")
        continue

    # Vars for window cropping
    acquisition_freq = result
    step = FRAME_INTERVAL // acquisition_freq    # e.g. 5 // 1 = 5 -> 1 image every 5 frames
    num_frames = MAX_TRACKING_DURATION // acquisition_freq
    # Adds + 1 if even to have the annotated pixel centered in the window
    target_size = WINDOW_SIZE # if WINDOW_SIZE%2 != 0 else WINDOW_SIZE + 1

    logger.debug("\tTime Info:")
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
    os.makedirs(os.path.join(BAD_CROPS, f'no_apo_{filename}'), exist_ok=True)

    # Load images to extract windows
    imgs = load_image_stack(os.path.join(IMG_DIR, f'{filename}.tif'))
    max_t, max_y, max_x = imgs.shape

    # Counter so we can sample same number of random tracks as apoptotic
    num_apo_crops = 0

    # Initialize column apoptotic with zeros
    merged_df_long['apoptotic'] = 0

    apo_track_ids = pd.DataFrame(columns=['track_id', 'apo_start_t'])

    apo_check_array = np.zeros_like(tracked_masks)

    logger.info("\tStarting cropping for apo cells")
    for i, row in tqdm(apo_annotations.iterrows(),
                       total=len(apo_annotations),
                       desc="Processing Annotations"):
        current_track_id = row.loc['matching_track']
        current_t = row.loc['t'] + row.loc['delta_ts']

        apo_track_ids.loc[i, 'track_id'] = current_track_id
        apo_track_ids.loc[i, 'apo_start_t'] = current_t

        # current_track_id = current_track_id.squeeze()
        if not np.isscalar(current_track_id):
            current_track_id = current_track_id.iloc[0]

        # Check if we found a match
        if current_track_id == 0:
            # If not we block a window
            annot_x = int(row['x'])
            annot_y = int(row['y'])
            annot_t = int(row['t'])

            # Block big windows if no match was found
            half_window = WINDOW_SIZE

            # Spatial indices (clamped to array bounds)
            x_start = max(0, annot_x - half_window)
            x_end = min(max_x, x_start + WINDOW_SIZE)  # Ensure window doesn't exceed array
            y_start = max(0, annot_y - half_window)
            y_end = min(max_y, y_start + WINDOW_SIZE)

            # Temporal indices (clamped)
            t_end = min(max_t, annot_t + NUM_BLOCKED_FRAMES//acquisition_freq)

            if t_end > last_t:
                apo_check_array[last_t:t_end, y_start:y_end, x_start:x_end] = 1


        is_correct_track = merged_df_long['track_id'] == current_track_id
        is_valid_time = merged_df_long['t'] >= current_t
        single_cell_df = merged_df_long.loc[is_correct_track & is_valid_time]
        if len(single_cell_df) < num_frames + 1:
            logger.warning(f"\t\tSkipping current track: {current_track_id}. "
                           f"Reason: Track lost too quickly after anno. "
                           f"len = {len(single_cell_df)}")
            continue

        last_row = single_cell_df.iloc[-1]
        last_x = int(last_row['x'])
        last_y = int(last_row['y'])
        last_t = int(last_row['t'])

        half_window = WINDOW_SIZE // 2

        # Spatial indices (clamped to array bounds)
        x_start = max(0, last_x - half_window)
        x_end = min(max_x, x_start + WINDOW_SIZE)  # Ensure window doesn't exceed array
        y_start = max(0, last_y - half_window)
        y_end = min(max_y, y_start + WINDOW_SIZE)

        # Temporal indices (clamped)
        t_end = min(max_t, last_t + NUM_BLOCKED_FRAMES//acquisition_freq)

        # Set the block to 1
        if t_end > last_t:
            apo_check_array[last_t:t_end, y_start:y_end, x_start:x_end] = 1
        

        # Count track length after manual apoptosis annotation (for histogram later)
        num_entries = single_cell_df.shape[0]
        survival_times.append(num_entries)

        # merged_df_long.loc[single_cell_df.index, 'apoptotic'] = 1
        is_correct_track = merged_df_long['track_id'] == current_track_id
        is_valid_time = merged_df_long['t'] >= current_t
        merged_df_long.loc[is_correct_track & is_valid_time, 'apoptotic'] = 1

        upper_t_limit = current_t + num_frames + step
        single_cell_df = single_cell_df.loc[single_cell_df['t'] < upper_t_limit]

        windows = []
        for _, sc_row in single_cell_df.iterrows():
            window = crop_window(imgs[int(sc_row['t'])],
                                 int(sc_row['x']),
                                 int(sc_row['y']),
                                 WINDOW_SIZE)
            if window.shape == (target_size, target_size):
                windows.append(window)
            else:
                windows.append(None)
                # print(f"{window.shape}: {target_size}")

        chosen_offset = None
        for offset in range(step):
            sub_windows = windows[offset::step]
            if all(x is not None for x in sub_windows):
                chosen_offset = offset
                break  # Stop as soon as a valid sequence is found

        if chosen_offset is None:
            logger.warning("\t\tNo valid start point for a sequence "
                           "with correct intervals found.")
        else:
            sub_windows = windows[chosen_offset::step]
            sub_windows = np.asarray(sub_windows)
            if len(sub_windows) == (MAX_TRACKING_DURATION // step) + 1:
                tiff.imwrite(
                    os.path.join(
                        CROPS_DIR,
                        filename,
                        f'trackID_{current_track_id}.tif'
                    ),
                    sub_windows.transpose(1, 2, 0)
                )
                tiff.imwrite(
                    os.path.join(
                        WINDOWS_DIR,
                        'apo',
                        f'apo_{filename}_{i}.tif'
                    ),
                    sub_windows.transpose(1, 2, 0)
                )
                num_apo_crops += 1
            else:
                logger.warning(f'\t\tAt least one of the images has the wrong '
                               'size after cropping and could not be used. '
                               'Length = {len(windows)}. Pos = {row["x"]}, '
                               f'{row["y"]}.')
    logger.info(f"\t\tValid crops of apo cells found for \
                {num_apo_crops}/{len(apo_annotations)}")
    
    tiff.imwrite(
        os.path.join(
            APO_CHECK_ARRAY_DIR,
            f'{filename}.tif'
        ),
        apo_check_array.transpose(1, 2, 0)
    )

    num_random_tracks = num_apo_crops

    logger.info('\tStarting cropping for non-apo cells.')
    # Get crops of healthy cells
    long_no_apo_df = merged_df_long[merged_df_long['apoptotic'] == 0]
    unique_track_ids = np.unique(long_no_apo_df['track_id'])

    num_healthy_crops = 0
    rejected_windows = 0

    for i, track_id in tqdm(enumerate(unique_track_ids),
                            total=len(unique_track_ids),
                            desc="Cropping non-apo Windows"):
        single_cell_df = long_no_apo_df.loc[
            long_no_apo_df['track_id'] == track_id
            ]
        start_t = min(single_cell_df['t'])
        # same length as for apoptotic cells
        single_cell_df = single_cell_df.loc[
            single_cell_df['t'] <= start_t + num_frames
            ]
        windows = []
        mask_windows = []
        for _, sc_row in single_cell_df.iterrows():
            window = crop_window(imgs[int(sc_row['t'])],
                                 int(sc_row['x']), int(sc_row['y']),
                                 WINDOW_SIZE)
            mask_window = crop_window(tracked_masks[int(sc_row['t'])], int(sc_row['x']), int(sc_row['y']), WINDOW_SIZE)  ###
            apo_window = crop_window(apo_check_array[int(sc_row['t'])], int(sc_row['x']), int(sc_row['y']), WINDOW_SIZE)
            
            mask_window = mask_window.astype(int)
            mask_window[mask_window != track_id] = 0
            # Check for blocked areas and window validity
            is_blocked = np.any(apo_window == 1)  # NEW: Check if blocked
            is_window_valid = (window.shape == (target_size, target_size)) and (not is_blocked)

            if is_window_valid:
                windows.append(window)
                mask_windows.append(mask_window)
            else:
                rejected_windows += 1
                logger.warning(
                    f"\t\tRejected window at (t={int(sc_row['t'])}, x={int(sc_row['x'])}, y={int(sc_row['y'])}): "
                    f"Blocked={is_blocked}, Shape={window.shape}"
                )

        if len(windows) == (MAX_TRACKING_DURATION) + 1:
            current_features = []
            for mask, img in zip(mask_windows, windows):    ###
                props = measure.regionprops_table(mask, img, properties=['label', 'eccentricity',
                                                                         'intensity_mean', 'intensity_std',
                                                                         'solidity', ])   ###
                feature_df = pd.DataFrame(props)
                
                if not feature_df.empty:
                    current_features.append(feature_df)
                # print(feature_df)
            if current_features:  # If any features were found for this track
                # Concatenate the features from this track into one DataFrame
                track_features = pd.concat(current_features, ignore_index=True)
                track_features['x'] = single_cell_df['x'].iloc[0]
                track_features['y'] = single_cell_df['y'].iloc[0]
                track_features['t'] = start_t
                mean_features = track_features.mean()
                mean_features['filename'] = f'cell_{filename}_{track_id}.tif'
                # Add to the overall list
                all_features.append(mean_features)

                mean_eccentricity = mean_features['eccentricity']
                mean_intensity = mean_features['intensity_mean']
                mean_std = mean_features['intensity_std']
                mean_solidity = mean_features['solidity']

                windows = np.asarray(windows)

                # Save all crops for features analysis
                tiff.imwrite(os.path.join(FEATURES_DIR, 'raw_images', f'cell_{filename}_{track_id}.tif'), windows[::step])
                tiff.imwrite(os.path.join(FEATURES_DIR, 'masks', f'cell_{filename}_{track_id}.tif'), mask_windows[::step])

                if any((mean_eccentricity < ECCENTRICITY_THR,
                    mean_std > CROP_STD_THR,
                    mean_intensity > CROP_MEAN_INT_THR,
                    mean_solidity < SOLIDITY_THR)):
                    tiff.imwrite(os.path.join(BAD_CROPS, f'no_apo_{filename}', f'trackID_{track_id}.tif'), windows[::step].transpose(1, 2, 0))
                    rejected_windows += 1
                else:
                    tiff.imwrite(
                        os.path.join(
                            CROPS_DIR,
                            f'no_apo_{filename}',
                            f'trackID_{track_id}.tif'
                        ),
                        windows[::step].transpose(1, 2, 0)
                    )
                    tiff.imwrite(
                        os.path.join(
                            WINDOWS_DIR,
                            'no_apo',
                            f'no_apo_{filename}_{i}.tif'
                        ),
                        windows[::step].transpose(1, 2, 0)
                    )
                    num_healthy_crops += 1


        else:
            logger.debug('\t\tAt least one of the images has the wrong size. '
                         f'Length = {len(windows)}. '
                         f'Pos = {sc_row["x"]}, {sc_row["y"]}.')
    logger.info(f"\t\tFound {num_healthy_crops} valid crops of healthy cells.")
    if rejected_windows > 0:
        logger.warning(f"\t\t{rejected_windows} windows were rejected "
                       "due to incorrect size.")

    # Gather random crops
    logger.info(f'\tStarting cropping for {num_random_tracks} random spots.')

    # Assuming imgs is a list/array of your image frames,
    # and each image has dimensions: img_height x img_width
    img_height, img_width = imgs[0].shape

    iter_count = 0
    crop_count = 0
    while (crop_count < num_random_tracks) and (iter_count <= 100):
        start_t = np.random.randint(0, len(imgs) - num_frames)

        # Randomly generate valid (x, y) coordinates for the crop center
        random_x = np.random.randint(WINDOW_SIZE//2, img_width - WINDOW_SIZE//2)
        random_y = np.random.randint(WINDOW_SIZE//2,img_height - WINDOW_SIZE//2)

        windows = []
        # Extract a window for each frame in the track duration
        for t in range(start_t, start_t + num_frames + 1, step):
            window = crop_window(imgs[t], random_x, random_y, WINDOW_SIZE)
            track_id_mask_crop = crop_window(tracked_masks[t],
                                             random_x, random_y,
                                             WINDOW_SIZE)
            apo_check_array_crop = crop_window(apo_check_array[t],
                                               random_x, random_y,
                                               WINDOW_SIZE)
            present_track_ids = track_id_mask_crop.flatten().tolist()

            # Continue here
            current_apo_track_ids = apo_track_ids.loc[
                apo_track_ids['apo_start_t'] <= t
                ]
            current_apo_ids = set(current_apo_track_ids['track_id'])

            is_pixel_not_in_apo = not any(pixel in current_apo_ids
                                          for pixel in present_track_ids)
            is_window_correct_size = window.shape == (target_size, target_size)
             # Check if the current cropped region is blocked (contains 1s in apo_check_array)
            is_area_blocked = np.any(apo_check_array_crop == 1)
            if (
                is_pixel_not_in_apo 
                and is_window_correct_size 
                and not is_area_blocked
            ):
                windows.append(window)
            else:
                if not is_pixel_not_in_apo:
                    logger.warning("\t\tCurrent window contains an apoptotic cell.")
                elif not is_window_correct_size:
                    logger.warning("\t\tWindow shape does not match target size.")
                elif is_area_blocked:
                    logger.warning("\t\tArea is in a blocked region.")
    

        if len(windows) == (MAX_TRACKING_DURATION//FRAME_INTERVAL) + 1:
            windows = np.asarray(windows)
            tiff.imwrite(
                os.path.join(
                    CROPS_DIR,
                    f'random_{filename}',
                    f'ID_{crop_count}.tif'
                ),
                windows.transpose(1, 2, 0)
            )
            tiff.imwrite(
                os.path.join(
                    WINDOWS_DIR,
                    'random',
                    f'random_{filename}_{crop_count}.tif'
                ),
                windows.transpose(1, 2, 0)
            )
            crop_count += 1
            iter_count += 1
        else:
            logger.warning(f'\t\tUnable to collect the desired number of '
                           f'images. Length windows = {len(windows)}')
            iter_count += 1

    logger.info(f"Finished processing {filename}: apo ({num_apo_crops}), \
                healthy ({num_healthy_crops}), random ({num_random_tracks})")


# Plotting
# Define the output directory
output_dir = os.path.join(PLOT_DIR, RUN_NAME)
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

# Create the plot
plt.figure(figsize=(8, 6))
plt.hist(
    survival_times,
    bins=range(
        min(survival_times),
        max(survival_times)+2
    ),
    edgecolor='black'
)
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

from scipy.stats import gaussian_kde


# Style parameters
plt.style.use('seaborn-v0_8-colorblind')


def plot_feature(data, feature_name, cutoff, xlim, comparison='>', units='', bins=20, output_dir='.'):
    """Enhanced plotting function with shaded regions and annotations"""
    text_xpos = 0.95  # Right-aligned annotations
    text_ha = 'right'
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate histogram
    counts, bins, patches = ax.hist(data, bins=bins, edgecolor='black', alpha=0.7)
    
    # Add KDE plot
    kde = gaussian_kde(data)
    x_vals = np.linspace(0, data.max(), 1000)
    ax.plot(x_vals, kde(x_vals)*len(data)*(bins[1]-bins[0]), color='#377eb8', linewidth=2)
    
    # Add cutoff line
    ax.axvline(cutoff, color='#e41a1c', linestyle='--', linewidth=2, label='Cutoff')
    
    if xlim is not None:
        ax.set_xlim(xlim)
        x_min, x_max = xlim
    else:
        x_min, x_max = ax.get_xlim()
    # Shade excluded region
    if comparison == '>':
        mask = data > cutoff
        shade_range = (cutoff, x_max)
        label = f'Excluded ({comparison} {cutoff}{units})'
    else:
        mask = data < cutoff
        shade_range = (x_min, cutoff)
        label = f'Excluded ({comparison} {cutoff}{units})'
        
    ax.axvspan(shade_range[0], shade_range[1], color='#e41a1c', alpha=0.2, label=label)
    
    # Calculate and annotate percentage
    excluded_pct = (mask.mean() * 100)
    ax.text(text_xpos, 0.95, f'{excluded_pct:.1f}% excluded',
            transform=ax.transAxes, ha=text_ha,
            bbox=dict(facecolor='white', alpha=0.8))
    
    # Formatting
    ax.margins(x=0)
    ax.set_xlabel(feature_name.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel('Number of Cells', fontsize=12)
    ax.set_title(f'{feature_name.replace("_", " ").title()} Distribution with Cutoff', fontsize=14)
    
    
    # Save and close
    plt.savefig(os.path.join(output_dir, f'{feature_name}_distribution.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()

features_df = pd.DataFrame(all_features)

features_df.to_csv(os.path.join(FEATURES_DIR, "features.csv"), header=True)

# Apply to all features with your cutoffs
plot_feature(features_df['eccentricity'], 'eccentricity', 0.4, xlim=(0,1),comparison='<', output_dir=output_dir)
plot_feature(features_df['intensity_std'], 'intensity_std', 1000, xlim=(0,3000), comparison='>', output_dir=output_dir)
plot_feature(features_df['solidity'], 'solidity', 0.925, xlim=(0.8,1), comparison='<', output_dir=output_dir)
plot_feature(features_df['intensity_mean'], 'intensity_mean', 3000, xlim=(0, 5500), comparison='>', output_dir=output_dir)

# Survival times plot (from your example)
plt.figure(figsize=(10, 6))
plt.hist(survival_times, bins=20, range=(0, 200), edgecolor='black', alpha=0.7)
plt.xlim(0, 200)
plt.xlabel("Survival Time (frames)", fontsize=12)
plt.ylabel("Number of Cells", fontsize=12)
plt.title("Histogram of Cell Survival Times (Frames)", fontsize=14)
plt.grid(True, alpha=0.4)
plt.savefig(os.path.join(output_dir, "survival_times_histogram.png"), 
           bbox_inches='tight', dpi=300)
plt.close()

