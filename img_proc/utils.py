import os
import subprocess
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm

import tifffile as tiff
from nd2reader import ND2Reader


# Plots
import matplotlib.pyplot as plt

# Segmentation specific
from csbdeep.utils import normalize
from skimage.morphology import remove_small_objects

# Tracking specific
import btrack
from btrack.constants import BayesianUpdates
from scipy.spatial.distance import cdist


logger = logging.getLogger(__name__)



def load_image_stack(path, target_channel_index):
    """
    Loads an image stack from a file based on its extension.

    Supports TIFF (.tif, .tiff) using `tifffile` and ND2 (.nd2) files using
    `ND2Reader`.

    Parameters
    ----------
    path : str
        The absolute path to the image stack file.

    Returns
    -------
    np.ndarray
        A NumPy array representing the loaded image stack. The shape will vary
        depending on the image dimensions (e.g., (T, H, W) for 2D+time).

    Raises
    ------
    ValueError
        If the file extension is not supported.
    FileNotFoundError
        If the specified file does not exist.
    """
    if path.endswith(('.tif', '.tiff')):
        # Load TIFF file using tifffile
        img_stack = tiff.imread(path)
    elif path.endswith('.nd2'):
        # Load ND2 file using ND2Reader and convert it to a numpy array
        with ND2Reader(path) as nd2:
            img_stack = np.array(nd2)
    else:
        raise ValueError(f"Unsupported file format for file: {path}")

    if img_stack.ndim <= 3:
        # If it's 3D (T, H, W) or less, it's single-channel
        return img_stack

    # 3. Channel Extraction Logic (Only if an index is provided and stack is multi-channel)
    if target_channel_index is not None:
        num_channels = img_stack.shape[1]
        
        # Input validation for the channel index
        if not (0 <= target_channel_index < num_channels):
            raise ValueError(
                f"Invalid channel index: {target_channel_index}. "
                f"Image stack has {num_channels} channels (indices 0 to {num_channels-1})."
            )
        single_channel_stack = img_stack[:,target_channel_index,:,:]
        return single_channel_stack
    
    # 4. If target_channel_index is None, return the full multi-channel stack
    return img_stack


def get_image_paths(directory):
    """
    Returns a sorted list of absolute paths for image files in a directory.

    Scans the specified directory for files with common image extensions
    (.tif, .tiff, .nd2, .npz) and returns their absolute paths, sorted
    alphabetically.

    Parameters
    ----------
    directory : str
        The path to the directory to scan for image files.

    Returns
    -------
    list of str
        A sorted list of absolute paths to the image files.

    Raises
    ------
    FileNotFoundError
        If the specified directory does not exist.
    """
    valid_extensions = ('.tif', '.tiff', '.nd2', '.npz')
    paths = [
        os.path.abspath(os.path.join(directory, f))
        for f in os.listdir(directory)
        if f.endswith(valid_extensions)
    ]
    return sorted(paths)

def get_experiment_info(filename, experiments_list):
    """
    Extracts specific experiment metadata from a DataFrame based on filename.

    Parses the filename to get an experiment identifier and then looks up
    corresponding details (magnification, acquisition/tracking frequencies,
    annotation frequency) from the provided `experiments_list` DataFrame.
    Returns default values if the experiment is not found.

    Parameters
    ----------
    filename : str
        The filename (e.g., "ExpXX_SiteXX") from which the experiment name
        (e.g., "ExpXX") will be extracted.
    experiments_list : pd.DataFrame
        DataFrame containing experiment parameters. Expected to have columns
        like 'Experiment', 'Magnification', 'Acquisition_frequency(min)',
        'Tracking_frequency', and 'Apo_annotation_frequency(min)'.

    Returns
    -------
    dict
        A dictionary containing experiment information. Keys include:
        'exp_name' (str): The extracted experiment name.
        'found' (bool): True if the experiment was found in `experiments_list`, False otherwise.
        'magnification' (str): Magnification (e.g., '40x', '20x'). Defaults to '40x'.
        'acquisition_freq' (float/int): Acquisition frequency in minutes.
        'tracking_freq' (float/int): Tracking frequency.
        'apo_annotation_freq' (float/int): Apoptotic annotation frequency in minutes.
        Values for frequencies will be `None` if the experiment is not found.
    """
    # Extract experiment name from filename
    exp_name = filename.split('_')[0]
    
    # Initialize the result dictionary with default values
    result = {
        'exp_name': exp_name,
        'found': False,
        'magnification': '40x',  # Default value
        'acquisition_freq': None,
        'tracking_freq': None,
        'apo_annotation_freq': None
    }
    
    # Find the experiment in the list
    exp_row = experiments_list[experiments_list['Experiment'] == exp_name]
    
    if exp_row.empty:
        logger.info(f"\t\tExperiment {exp_name} not found in experiment info.")
    else:
        # Update result with values from the experiment list
        result['found'] = True
        result['magnification'] = exp_row['Magnification'].values[0]
        result['acquisition_freq'] = exp_row['Acquisition_frequency(min)'].values[0]
        result['tracking_freq'] = exp_row['Tracking_frequency'].values[0]
        result['apo_annotation_freq'] = exp_row['Apo_annotation_frequency(min)'].values[0]
        
        logger.info(f"\t\tFound experiment {exp_name} in experiment info.")
    
    return result


def sync_scratch_dirs(config: dict):
    """
    Sync directories from scratch to final destination if SCRATCH_DIR is defined.

    Parameters
    ----------
    config : dict
        Dictionary containing paths for scratch and final directories.
        Must contain 'SCRATCH_DIR' (can be None) and the relevant crop/output directories.
    """
    scratch_base = config.get('SCRATCH_DIR')
    base_path = config.get('BASE_DATA_DIR')
    if scratch_base is None:
        logger.info("No scratch directory defined; skipping sync.")
        return

    # Ensure the final destination base path exists
    final_base_path = Path(base_path)
    final_base_path.mkdir(parents=True, exist_ok=True)

    dirs_to_sync = [
        ('UPSAMPLE_DIR', 'upsampled crops'),
        ('WINDOWS_DIR', 'base crops'),
        ('WINDOWS_DIR_20X', '20X crops')
    ]

    for key, desc in dirs_to_sync:
        # Define paths for the directory and the archive
        scratch_dir_path = Path(scratch_base) / Path(config[key]).name
        archive_name = f"{Path(config[key]).name}.tar.xz" # Using .tar.xz for best lossless compression
        scratch_archive_path = Path(scratch_base) / archive_name
        final_archive_path = final_base_path / archive_name
        
        if not scratch_dir_path.exists():
            logger.warning(f"Scratch path {scratch_dir_path} does not exist; skipping {desc}.")
            continue

        # --- A. COMPRESS on Scratch ---
        logger.info(f"A. Compressing {desc} directory on scratch: {scratch_dir_path}")
        try:
            # Command 1: tar -cf - -C <parent_dir> <target_dir> (creates uncompressed archive to stdout)
            # Command 2: pigz -p <cores> > <archive_path> (reads stdin, compresses, writes to file)
            
            tar_command = ["tar", "-cf", "-", "-C", str(scratch_dir_path.parent), scratch_dir_path.name]
            pigz_command = ["pigz", "-p", "10", ">", str(scratch_archive_path)] # Using 10 cores
            # TODO: add global constant for num cores during compression

            # We use shell=True here to properly handle the pipe, and combine the commands
            subprocess.run(
                f"{' '.join(tar_command)} | {' '.join(pigz_command)}",
                check=True, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            logger.info(f"Successfully created archive with pigz: {scratch_archive_path.name}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error compressing {desc} on scratch: {e}")
            continue

        # --- B. SYNC the Archive (Single large block transfer) ---
        logger.info(f"B. Syncing single archive block to final destination: {final_base_path}")
        try:
            # Command: rsync -aP /scratch/.../archive.tar.xz /perm_storage/.../
            subprocess.run(
                ["rsync", "-aP", str(scratch_archive_path), str(final_base_path)],
                check=True
            )
            logger.info(f"Finished syncing archive: {scratch_archive_path.name}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error syncing archive {desc}: {e}")
            # Do not proceed to clean up if sync failed
            continue

        # --- C. DECOMPRESS on Final Destination ---
        logger.info(f"C. Decompressing archive on final destination: {final_base_path}")
        try:
            # Command 1: unpigz -p <cores> <archive_path> (reads file, decompresses to stdout)
            # Command 2: tar -xf - -C <final_base_path> (reads uncompressed data from stdin, extracts)
            
            unpigz_command = ["unpigz", "-p", "4", "-c", str(final_archive_path)] # -c: write to stdout
            tar_command = ["tar", "-xf", "-", "-C", str(final_base_path)] # -f -: read from stdin

            subprocess.run(
                f"{' '.join(unpigz_command)} | {' '.join(tar_command)}",
                check=True, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            logger.info(f"Finished decompressing with unpigz: {desc}.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error decompressing {desc} on final destination: {e}")
            continue

        # --- D. Clean up Archives on Scratch and Final ---
        # Cleanup scratch directory after successful sync/decompression
        try:
            shutil.rmtree(scratch_dir_path)
            scratch_archive_path.unlink(missing_ok=True)
            final_archive_path.unlink(missing_ok=True)
            logger.info(f"Cleaned up scratch directory and all archive files for {desc}.")
        except Exception as e:
            logger.error(f"Error cleaning up files for {desc}: {e}")


# --- Segmentation specific functions ---


def run_segmentation(imgs, model, axis_norm):
    """
    Performs instance segmentation on an image stack.

    Loads an image stack, normalizes each image, and then applies a provided
    segmentation model to predict instance masks and their details.

    Parameters
    ----------
    imgs : np.ndarray
        Path to the image stack file (e.g., a TIFF stack).
    model
        A segmentation model object (e.g., StarDist model) with a
        `predict_instances` method.
    axis_norm : tuple or None
        Axes along which to normalize the images (e.g., (0,1) for 2D,
        None for global). Passed directly to the `normalize` function.

    Returns
    -------
    gt : np.ndarray
        A 3D NumPy array of segmented instance masks (time, height, width),
        where each non-zero pixel represents an object ID.
    details : list of dict
        A list of dictionaries, one per frame, containing prediction details
        from the segmentation model (e.g., centroids, probabilities).
    """
    h2b_imgs_normal = np.asarray([normalize(img, 1, 99.8, axis=axis_norm) for img in imgs])
    
    gt = []
    details = []
    for x in tqdm(h2b_imgs_normal, desc="Segmenting"):
        labels, det = model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False)
        gt.append(labels)
        details.append(det)
    gt = np.asarray(gt)
    return gt, details  

def filter_segmentation(gt, details, min_size):
    """
    Filters segmented objects by size and creates a summary DataFrame.

    Removes objects from segmentation masks that are smaller than a specified
    minimum size. For the remaining objects, it extracts their unique IDs,
    time points, and centroid coordinates, compiling this information into
    a pandas DataFrame.

    Parameters
    ----------
    gt : np.ndarray
        A 3D NumPy array of segmented instance masks (time, height, width).
        Each non-zero pixel represents an object ID.
    details : list of dict
        A list of dictionaries, one per frame, containing prediction details
        from the segmentation model. Expected to have a 'points' key
        with object centroid coordinates.
    min_size : int
        The minimum pixel area for an object to be retained. Objects with
        fewer pixels than this size will be removed.

    Returns
    -------
    gt_filtered : np.ndarray
        A copy of the input `gt` array with objects smaller than `min_size`
        removed (i.e., their pixels set to zero). Data type is `np.uint16`.
    summary_df : pd.DataFrame
        A DataFrame summarizing the filtered objects with columns:
        'obj_id' (int): Original unique ID of the object.
        't' (int): Time frame index.
        'x' (float): X-coordinate (column index) of the object's centroid.
        'y' (float): Y-coordinate (row index) of the object's centroid.
    """
    logger.info(f"\tRemoving objects smaller than {min_size}.")
    num_frames = gt.shape[0]
    gt_filtered = np.zeros_like(gt, dtype=np.uint16)
    df_list = []
    for frame in range(num_frames):
        gt_filtered[frame] = remove_small_objects(gt[frame], min_size=min_size)
        unique_ids = np.unique(gt_filtered[frame])
        unique_ids = unique_ids[unique_ids > 0]
        x, y = [], []
        timepoint = np.full_like(unique_ids, frame)
        current_details = details[frame]['points']
        for obj_id in unique_ids:
            position = current_details[obj_id - 1]  # Adjust indexing as needed
            x.append(position[1])
            y.append(position[0])
        current_df = pd.DataFrame({'obj_id': unique_ids, 't': timepoint, 'x': x, 'y': y})
        df_list.append(current_df)
    summary_df = pd.concat(df_list, ignore_index=True)
    logging.info("\t\tDone!")
    return gt_filtered, summary_df


# --- Tracking specific functions ---
def get_btrack_params(filename, experiments_list, config_params):
    """
    Determines the appropriate btrack configuration file and search radius 
    based on experiment parameters.
    
    Args:
        config_params: A dictionary containing all relevant tracking config values 
                       (BT_CONFIG_FILE, BT_CONFIG_20X, EPS_TRACK, etc.).
    """
    exp_name = filename.split('_')[0]
    exp_row = experiments_list[experiments_list['Experiment'] == exp_name]
    
    default_config_file = config_params['BT_CONFIG_FILE']
    default_track_radius = config_params['EPS_TRACK']
    
    if exp_row.empty:
        logger.warning(f"\tNo matching experiment found for {exp_name}, using default config")
        return default_config_file, default_track_radius
    
    # Extract values safely
    acq_freq = exp_row['Acquisition_frequency(min)'].values[0]
    magnification = exp_row['Magnification'].values[0]

    # Logic to select specialized config
    if acq_freq == 1 and magnification == '20x':
        logger.info(f"\tUsing 20x t1 config for {filename}")
        return config_params['BT_CONFIG_20X'], config_params['EPS_TRACK_20x']
    elif acq_freq == 5 and magnification == '20x':
        logger.info(f"\tUsing 20x t5 config for {filename}")
        return config_params['BT_CONFIG_20X_5t'], config_params['EPS_TRACK_20x']
    else:
        logger.info(f"\tUsing standard config for {filename}")
        return default_config_file, default_track_radius

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


def plot_track_lengths(merged_df, min_len, filename, output_dir, run_name):
    """
    Generates and saves track length histograms (original and filtered).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plot_path = os.path.join(output_dir, run_name, "btrack_hists")
    os.makedirs(plot_path, exist_ok=True)

    # 1. Full Track Lengths
    track_lengths = merged_df["track_id"].value_counts()
    plt.figure(figsize=(10, 6))
    plt.hist(track_lengths, bins=20, edgecolor="black", alpha=0.7)
    plt.xlabel("Track Length (frames)")
    plt.ylabel("Frequency")
    plt.title(f"Track Lengths (All Tracks) for {filename}")
    plt.savefig(os.path.join(plot_path, f"{filename}_track_lengths.png"))
    plt.close()
    logger.info(f"\t\tSaved track length histogram: {filename}_track_lengths.png")

    # 2. Filtered Track Lengths
    track_sizes = merged_df.groupby("track_id")["track_id"].transform('size')
    merged_df_long = merged_df[track_sizes >= min_len].copy()
    track_lengths_long = merged_df_long["track_id"].value_counts()
    
    plt.figure(figsize=(10, 6))
    plt.hist(track_lengths_long, bins=20, edgecolor="black", alpha=0.7)
    plt.xlabel("Track Length (frames)")
    plt.ylabel("Frequency")
    plt.title(f"Track Lengths (Tracks >= {min_len} frames) for {filename}")
    plt.savefig(os.path.join(plot_path, f"{filename}_track_lengths_long.png"))
    plt.close()
    logger.info(f"\t\tSaved filtered track length histogram: {filename}_track_lengths_long.png")


def fill_track_gaps_vectorized(df, distance_threshold=50):
    """
    Fills NaN track_ids processing frame-by-frame to avoid OOM.
    Uses float32 and scipy cdist for speed and lower memory usage.
    """
    # Early exit if no gaps
    is_gap = df['track_id'].isna()
    if not is_gap.any():
        return df
    
    
    # Get frames with gaps
    gap_times = df.loc[is_gap, 't'].unique()
    
    # Pre-filter candidates (only keep rows with valid track_ids)
    candidates_df = df.loc[~is_gap, ['x', 'y', 't', 'track_id']].copy()
    
    # Store assignments
    new_assignments = {}
    
    # Process each frame with gaps
    for t in gap_times:
        # Get gaps at time t
        current_gaps = df.loc[(df['t'] == t) & is_gap]
        
        # Get candidates from adjacent frames
        current_candidates = candidates_df[candidates_df['t'].isin([t - 1, t + 1])]
        
        if current_candidates.empty:
            continue
        
        # Vectorized distance calculation
        gap_coords = current_gaps[['x', 'y']].values
        cand_coords = current_candidates[['x', 'y']].values
        dists = cdist(gap_coords, cand_coords, metric='euclidean')
        
        # Find nearest neighbor for each gap
        min_idxs = np.argmin(dists, axis=1)
        min_dists = dists[np.arange(len(gap_coords)), min_idxs]
        
        # Apply threshold and store assignments
        valid_mask = min_dists < distance_threshold
        gap_indices = current_gaps.index[valid_mask]
        matched_track_ids = current_candidates.iloc[min_idxs[valid_mask]]['track_id'].values
        
        for idx, track_id in zip(gap_indices, matched_track_ids):
            new_assignments[idx] = track_id
    
    # Apply all assignments at once
    if new_assignments:
        assignment_series = pd.Series(new_assignments)
        df.loc[assignment_series.index, 'track_id'] = assignment_series.values
    
    return df


# --- Matching specific funcitons ---
def check_temporal_compatibility(
        filename: str,
        experiments_df: pd.DataFrame,
        target_interval: int
    ) -> tuple[bool, str | int]:
    """
    Checks compatibility between experiment acquisition frequency and target interval.

    Extracts the acquisition frequency for a given experiment from a DataFrame
    and validates if it's compatible with a specified target interval.

    Parameters
    ----------
    filename : str
        The filename (e.g., "ExpXX_SiteXX") from which the experiment ID is extracted.
    experiments_df : pd.DataFrame
        DataFrame containing experiment parameters, expected to have an
        'Experiment' column and an 'Acquisition_frequency(min)' column.
    target_interval : int
        The desired target interval in minutes for processing.

    Returns
    -------
    tuple[bool, str | int]
        A tuple indicating validity and result:
        - (True, acquisition_freq: int) if compatible.
        - (False, error_message: str) if incompatible, with a reason.
    """
    # The implementation for this function is kept exactly as provided by the user
    experiment_id = filename.split('_')[0]
    matching_row = experiments_df[experiments_df['Experiment'] == experiment_id]

    if matching_row.empty:
        return False, f"Experiment {experiment_id} not found in registry"
    
    try:
        acquisition_freq = int(matching_row['Acquisition_frequency(min)'].iloc[0])
    except (ValueError, TypeError):
        return False, f"Invalid frequency format in registry (file: {filename})"
    
    if acquisition_freq <= 0:
        return False, f"Invalid acquisition frequency: {acquisition_freq} (file: {filename})"
    if target_interval % acquisition_freq != 0:
        return False, f"{target_interval}min target not divisible by {acquisition_freq}min acquisition (file: {filename})"

    return True, acquisition_freq


def match_annotations(apo_annotations, details, tracked_masks, gt_filtered, dt_annots):
    """
    Matches manual annotations with segmented and tracked objects.

    This function attempts to find corresponding automated detections and tracks
    for each manual annotation point. It uses a multi-stage matching strategy:
    direct pixel lookup, nearest centroid, and a temporal neighborhood search.

    Parameters
    ----------
    apo_annotations : pd.DataFrame
        DataFrame containing manual annotations. Expected columns:
        't' (time frame), 'x' (x-coordinate), 'y' (y-coordinate).
    details : list of dict
        List of dictionaries, one per time point, containing segmentation
        details. Each dictionary is expected to have a 'points' key,
        where `details[t]['points']` is a list/array of (y, x) centroids
        for objects in frame `t`.
    tracked_masks : np.ndarray
        A 3D NumPy array (time, height, width) where non-zero pixels
        contain unique btrack `track_id`s.
    gt_filtered : np.ndarray
        A 3D NumPy array (time, height, width) where non-zero pixels
        contain unique `obj_id`s after filtering. Used for getting `obj_id`
        at a matched location.
    dt_annots : int
        The time resolution of manual annotations in minutes per frame.
        Used to synchronize annotation time points with image frames.

    Returns
    -------
    pd.DataFrame
        The `apo_annotations` DataFrame updated with new columns:
        - 'matching_object' (int): The `obj_id` of the matched segmented object.
        - 'matching_track' (int): The `track_id` of the matched trajectory.
        - 'strdst_x' (float): X-coordinate of the matched object's centroid.
        - 'strdst_y' (float): Y-coordinate of the matched object's centroid.
        - 'delta_ts' (int): Temporal offset (in frames) from the annotation's
          frame where a match was found (0 if found in the same frame).
    dict
        A dictionary containing evaluation metrics:
        - 'num_matches' (int): Count of annotations successfully matched.
        - 'num_mismatches' (int): Count of annotations with no suitable match.
        - 'dist_paolo_stardist' (list): Euclidean distance between manual
          annotation and the centroid of the finally matched object.
        - 'dist_alt_matching' (list): Euclidean distance between manual
          annotation and the closest detected centroid in the same frame.

    Notes
    -----
    - Assumes `details[t]['points']` provides (y, x) coordinates for centroids.
    - Assumes `obj_id`s in `gt_filtered` are 1-based for indexing into `details[t]['points']`.
    - Manual annotation time ('t') is adjusted: `(t-1) * dt_annots` to align
      with 0-indexed segmentation frames.
    - The temporal search window for a match is fixed to +3 frames.
    """
    delta_ts = []
    corresponding_objs = []
    corresponding_tracks = []
    strdst_x = []
    strdst_y = []
    dist_paolo_stardist = []
    dist_alt_matching = []
    num_matches = 0
    num_mismatches = 0

    # Loop over each annotation
    for _, row in tqdm(apo_annotations.iterrows(), total=len(apo_annotations), desc="Processing Annotations"):
        t, x, y = int(row['t']), int(row['x']), int(row['y'])
        # -1 because of 1 indexing in annotations!
        t = (t - 1) * dt_annots
        centroids = np.array(details[t]['points'])  # Convert list to NumPy array
        
        # Compute Euclidean distances
        distances = np.linalg.norm(centroids - np.array([y, x]), axis=1)
        match_index = np.argmin(distances)
        alt_distance_to_paolo = distances[match_index]
        dist_alt_matching.append(alt_distance_to_paolo)

        # Check for a perfect match using the mask values
        if tracked_masks[t, y, x] != 0:
            corresponding_tracks.append(tracked_masks[t, y, x])
            corresponding_objs.append(gt_filtered[t, y, x])
            delta_ts.append(0)
            num_matches += 1
        else:
            t_start = t
            t_end = min(t + 3 + 1, tracked_masks.shape[0])
            previous_frames = tracked_masks[t_start:t_end, y, x]
            if previous_frames.size > 0:
                non_zero_frames = previous_frames[previous_frames != 0]
                if non_zero_frames.size > 0:
                    counts = np.bincount(non_zero_frames)
                    match_id = np.argmax(counts)
                else:
                    match_id = 0
            else:
                match_id = 0

            if match_id == 0:
                num_mismatches += 1
            else:
                num_matches += 1

            # Search adjacent frames for a matching track
            for delta in [0, 1, 2, 3]:
                if (t + delta) in range(t_start, t_end):
                    if tracked_masks[t + delta, y, x] == match_id:
                        corresponding_tracks.append(match_id)
                        corresponding_objs.append(gt_filtered[t + delta, y, x])
                        delta_ts.append(delta)
                        break

        # Extract centroid of the best matching object
        centroids = np.array(details[t + delta_ts[-1]]['points'])
        match_centroid = centroids[corresponding_objs[-1] - 1] if corresponding_objs[-1] > 0 else (9999, 9999)
        strdst_x.append(match_centroid[1])
        strdst_y.append(match_centroid[0])
        # Compute distance for validation or plotting
        distance_to_paolo = np.sqrt(((match_centroid[1] - row['x']) ** 2 + (match_centroid[0] - row['y']) ** 2))
        dist_paolo_stardist.append(distance_to_paolo)

    # Update the apo_annotations DataFrame
    apo_annotations['matching_object'] = corresponding_objs
    apo_annotations['matching_track'] = corresponding_tracks
    apo_annotations['strdst_x'] = strdst_x
    apo_annotations['strdst_y'] = strdst_y
    apo_annotations['delta_ts'] = delta_ts

    metrics = {
        'num_matches': num_matches,
        'num_mismatches': num_mismatches,
        'dist_paolo_stardist': dist_paolo_stardist,
        'dist_alt_matching': dist_alt_matching
    }

    return apo_annotations, metrics


def plot_matching_distances(metrics_list, output_dir, run_name):
    """
    Generates and saves histograms comparing matching distances.

    Args:
        metrics_list (list): A list of dictionaries, where each dict contains
                             'dist_paolo_stardist' and 'dist_alt_matching' arrays.
        output_dir (str): Base output directory for plots.
        run_name (str): Name used for plotting subdirectories.
    """
    output_path = os.path.join(output_dir, run_name)
    os.makedirs(output_path, exist_ok=True) 

    # Flatten the distance lists from all files
    dist_paolo_stardist_flat = np.concatenate([m['dist_paolo_stardist'] for m in metrics_list]).tolist()
    dist_alt_matching_flat = np.concatenate([m['dist_alt_matching'] for m in metrics_list]).tolist()

    # --- Plot 1: Custom Matching Distances ---
    plt.figure(figsize=(8, 6))
    plt.hist(dist_paolo_stardist_flat, bins=25, range=(0, 50), color='blue', edgecolor='black', alpha=0.7)
    plt.xlim(0, 50)
    plt.xlabel('Distance (L2-norm, Pixels)')
    plt.ylabel('Frequency')
    plt.title('Distances between Manual Annotation and Matched Centroid (Custom)')
    plt.savefig(os.path.join(output_path, "distances_histogram.png"))
    plt.close()

    # --- Plot 2: Matching Comparison ---
    plt.figure(figsize=(10, 5))
    plt.hist(dist_paolo_stardist_flat, bins=30, alpha=0.5, label="Custom Matching", color='blue', range=(0, 60))
    plt.hist(dist_alt_matching_flat, bins=30, alpha=0.5, label="L2 Norm Matching (Closest Centroid)", color='red', range=(0, 60))
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.title("Comparison of Matching Approaches")
    plt.legend()
    plt.savefig(os.path.join(output_path, "matching_comparison_histogram.png"))
    plt.close()


# --- Cropping specific functions ---
import numpy as np

def crop_window(img, center_x, center_y, window_size):
    """
    Crops a square window from a 2D image around a specified center.

    The window's boundaries are clamped to the image dimensions, so it will
    not extend beyond the image edges.

    Parameters
    ----------
    img : np.ndarray
        The 2D image array to crop from.
    center_x : int
        The X-coordinate (column index) of the desired window center.
    center_y : int
        The Y-coordinate (row index) of the desired window center.
    window_size : int
        The desired side length of the square window in pixels.

    Returns
    -------
    np.ndarray
        The cropped 2D image window.
    """
    # Check if number is even, add one if so
    # if window_size%2 == 0:
    #     window_size += 1
    half_window_size = window_size // 2
    x_from = max(center_x - half_window_size, 0)
    x_to = min(center_x + half_window_size, img.shape[1])
    y_from = max(center_y - half_window_size, 0)
    y_to = min(center_y + half_window_size, img.shape[0])
    window = img[y_from:y_to, x_from:x_to]

    return window

def block_window_in_array(array, t, x, y, window_size, num_blocked_frames, acquisition_freq):
    """
    Blocks a window in the given array around the specified coordinates.
    
    Args:
        array: The 3D array to block in (t, y, x format)
        t: Time index
        x: X coordinate
        y: Y coordinate
        window_size: Size of the window
        num_blocked_frames: Number of frames to block forward in time
        acquisition_freq: Acquisition frequency for scaling time blocks
    """
    max_t, max_y, max_x = array.shape
    half_window = window_size // 2
    
    # Spatial indices (clamped to array bounds)
    x_start = max(0, x - half_window)
    x_end = min(max_x, x_start + window_size)
    y_start = max(0, y - half_window)
    y_end = min(max_y, y_start + window_size)
    
    # Temporal indices (clamped)
    t_end = min(max_t, t + num_blocked_frames // acquisition_freq)
    
    # Set the block to 1 if there's a valid time range
    if t_end > t:
        array[t:t_end, y_start:y_end, x_start:x_end] = 1