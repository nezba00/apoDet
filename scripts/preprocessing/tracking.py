import logging
import pandas as pd
import numpy as np

import trackpy as tp
import btrack
from btrack.constants import BayesianUpdates
from .config import TRACKING_CONFIG

logger = logging.getLogger(__name__)

def remove_outlier_frames(label_stack, thr_multiplier = 20):
    bin_imgs = np.copy(label_stack)
    bin_imgs[label_stack != 0] = 1

    min_differences = []
    for i, img in enumerate(bin_imgs):
        if i == 0:
            diff = np.abs(img - bin_imgs[i + 1])
            sad = np.sum(diff)
            min_differences.append(sad)
        elif i == len(bin_imgs)-1:
            diff = np.abs(bin_imgs[i - 1] - img)
            sad = np.sum(diff)
            min_differences.append(sad)
        else:
            diff1 = np.abs(img - bin_imgs[i + 1])
            diff2 = np.abs(img - bin_imgs[i - 1])
            sad1 = np.sum(diff1)
            sad2 = np.sum(diff2)
    
            min_sad = min(sad1, sad2)
            min_differences.append(min_sad)
    # Convert to numpy arrays
    min_differences = np.array(min_differences)

    median = np.median(min_differences)
    mad = np.median(np.abs(min_differences - median))
    threshold = median + thr_multiplier * mad

    # Find outlier indices
    outlier_indices = np.where(min_differences > threshold)[0]
    
    # Replace outlier frames with zero
    cleaned_stack = np.copy(label_stack)
    for idx in outlier_indices:
        cleaned_stack[idx] = np.zeros_like(label_stack[idx])
    
    return cleaned_stack, outlier_indices

def run_tracking(gt_filtered, bt_config_file, track_radius):
    logger.info("\tStarting tracking")
    _, fovY, fovX = gt_filtered.shape
    btObj = btrack.utils.segmentation_to_objects(gt_filtered, properties=("area",), assign_class_ID=True)
    
    with btrack.BayesianTracker() as tracker:
        tracker.configure(bt_config_file)
        tracker.update_method = BayesianUpdates.APPROXIMATE
        tracker.max_search_radius = track_radius
        tracker.append(btObj)
        tracker.volume = ((0, fovX), (0, fovY))
        tracker.track(step_size=100)
        tracker.optimize()
        btTracks = tracker.tracks
    
    dfBTracks = pd.concat(pd.DataFrame(t.to_dict(["ID", "t", "x", "y"])) for t in btTracks)
    dfBTracks.rename(columns={"ID": "track_id", "t": "t", "x": "x", "y": "y", "class_id": "obj_id"}, inplace=True)
    dfBTracks["obj_id"] = dfBTracks["obj_id"].astype("Int32")
    logger.info("\t\tTracking Done.")
    return dfBTracks


def convert_obj_to_track_ids(gt_filtered, merged_df):
    """
    Converts object IDs to track IDs in Stardist masks using tracking data.

    Parameters:
        gt_filtered (np.ndarray): The 3D array (time, height, width) of segmentation masks.
        merged_df (pd.DataFrame): DataFrame containing tracking information with 'obj_id', 'track_id', and 't'.

    Returns:
        np.ndarray: New 3D mask where obj_ids are replaced with track_ids.
    """
    
    tracked_masks = np.zeros_like(gt_filtered)

    for t, mask_frame in enumerate(gt_filtered):
        current_df = merged_df[merged_df['t'] == t]

        # Create a mapping {obj_id: track_id} for this timepoint
        obj_to_track = current_df.set_index('obj_id')['track_id'].to_dict()

        # Replace obj_id in the mask with the corresponding track_id
        for obj_id, track_id in obj_to_track.items():
            if pd.isna(track_id):
                continue
            
            tracked_masks[t][mask_frame == obj_id] = track_id

    return tracked_masks


def get_btrack_config_path(filename, experiments_list):
    """
    Determine which BTrack config file to use based on experiment parameters.
    
    Parameters:
        filename (str): The filename being processed (format: ExpXX_SiteXX)
        experiments_list (pd.DataFrame): DataFrame with experiment parameters
    
    Returns:
        str: Path to the appropriate BTrack config file
        bool: Whether special settings were applied
    """
    # Extract experiment name from filename
    exp_name = filename.split('_')[0]  # This will give "ExpXX"
    
    # Find matching experiment in the list
    exp_row = experiments_list[experiments_list['Experiment'] == exp_name]
    
    if exp_row.empty:
        logger.warning(f"\tNo matching experiment found for {exp_name}, using default config")
        return TRACKING_CONFIG['BT_CONFIG_FILE'], False
    
    # Check if this experiment needs the special config
    if (exp_row['Acquisition_frequency(min)'].values[0] == 1 and 
        exp_row['Magnification'].values[0] == '20x'):
        logger.info(f"\tUsing 20x config for {filename} (matched experiment: {exp_name})")
        return TRACKING_CONFIG['BT_CONFIG_20X'], True
    else:
        logger.info(f"\tUsing standard config for {filename} (matched experiment: {exp_name})")
        return TRACKING_CONFIG['BT_CONFIG_FILE'], False

