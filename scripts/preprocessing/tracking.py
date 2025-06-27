import logging
import pandas as pd
import numpy as np

import trackpy as tp
import btrack
from btrack.constants import BayesianUpdates
from .config import TRACKING_CONFIG

logger = logging.getLogger(__name__)

def remove_outlier_frames(label_stack, thr_multiplier = 20):
    """
    Removes temporally isolated segmentation-mask frames that are statistical outliers.

    Identifies and zeroes out "glitchy" frames in a time-series of segmentation
    masks (e.g., video or volumetric slices) by comparing each frame's
    sum-of-absolute-differences (SAD) to its nearest neighbor against a
    threshold derived from the median absolute deviation (MAD) of all SADs.

    Parameters
    ----------
    label_stack : np.ndarray
        A 3D array (T, H, W) or (T, H, W, C) containing integer labels per frame.
        Zero is background, non-zero is foreground.
    thr_multiplier : float, optional
        Multiplier for the MAD to define the outlier threshold.
        Frames with SAD to neighbors exceeding `median(SADs) + thr_multiplier * MAD(SADs)`
        are zeroed out. Defaults to 20.0.

    Returns
    -------
    cleaned_stack : np.ndarray
        A copy of 'label_stack' with detected outlier frames replaced by all zeros.
    outlier_indices : np.ndarray
        1D integer array listing the indices of frames identified as outliers.
    """
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
    """
    Performs object tracking on segmented data using the btrack library.

    Processes time-series segmentation data, configures a Bayesian tracker,
    and returns the resulting object tracks as a pandas DataFrame.

    Parameters
    ----------
    gt_filtered : np.ndarray
        3D NumPy array (num_frames, height, width) of segmented objects.
    bt_config_file : str or PathLike
        Path to the btrack configuration file (e.g., JSON).
    track_radius : float
        Maximum search radius (pixels) for linking objects between frames.

    Returns
    -------
    pandas.DataFrame
        DataFrame with tracked object data: 'track_id', 't', 'x', 'y'.
        Includes 'obj_id' (original object ID) if 'class_id' is
        included in the `to_dict` call during conversion.

    Notes
    -----
    - Assumes 2D + time input; adjust `tracker.volume` for 3D + time.
    """
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
    Replaces segmentation object IDs with persistent track IDs in masks.

    Iterates through time frames of segmentation masks and uses tracking
    information to assign a consistent `track_id` to each segmented object
    across frames, enabling visualization and analysis of trajectories.

    Parameters
    ----------
    gt_filtered : np.ndarray
        3D NumPy array (time, height, width) of segmentation masks.
        Non-zero pixels within a frame correspond to objects with unique
        integer `obj_id`s.
    merged_df : pd.DataFrame
        DataFrame with tracking results. Must contain 't', 'obj_id', and
        'track_id' columns.

    Returns
    -------
    np.ndarray
        New 3D NumPy array of the same shape as `gt_filtered`, where
        original `obj_id`s are replaced by their corresponding `track_id`s.
        Background (zero) pixels remain zero.
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
    Determines the appropriate btrack configuration file and search radius.

    Selects a btrack configuration file and corresponding search radius based
    on experiment parameters (acquisition frequency and magnification) found
    in `experiments_list` for a given `filename`. Uses default settings if
    no specific match is found.

    Parameters
    ----------
    filename : str
        The filename being processed (e.g., "ExpXX_SiteXX"), used to extract
        the experiment identifier.
    experiments_list : pd.DataFrame
        DataFrame containing experiment parameters. Expected to have 'Experiment',
        'Acquisition_frequency(min)', and 'Magnification' columns.

    Returns
    -------
    str
        Path to the appropriate btrack configuration file.
    int
        Search radius in pixels for tracking.
    """
    # Extract experiment name from filename
    exp_name = filename.split('_')[0]  # This will give "ExpXX"
    
    # Find matching experiment in the list
    exp_row = experiments_list[experiments_list['Experiment'] == exp_name]
    
    if exp_row.empty:
        logger.warning(f"\tNo matching experiment found for {exp_name}, using default config")
        return TRACKING_CONFIG['BT_CONFIG_FILE'], TRACKING_CONFIG['EPS_TRACK']
    
    # Check if this experiment needs the special config
    if (exp_row['Acquisition_frequency(min)'].values[0] == 1 and 
        exp_row['Magnification'].values[0] == '20x'):
        logger.info(f"\tUsing 20x t1 config for {filename} (matched experiment: {exp_name})")
        return TRACKING_CONFIG['BT_CONFIG_20X'], TRACKING_CONFIG['EPS_TRACK_20x']
    elif (exp_row['Acquisition_frequency(min)'].values[0] == 5 and 
        exp_row['Magnification'].values[0] == '20x'):
        logger.info(f"\tUsing 20x t5 config for {filename} (matched experiment: {exp_name})")
        return TRACKING_CONFIG['BT_CONFIG_20X_5t'], TRACKING_CONFIG['EPS_TRACK_20x']
    else:
        logger.info(f"\tUsing standard config for {filename} (matched experiment: {exp_name})")
        return TRACKING_CONFIG['BT_CONFIG_FILE'], TRACKING_CONFIG['EPS_TRACK']

