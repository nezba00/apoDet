import pandas as pd
from tqdm import tqdm
import numpy as np







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
    - Manual annotation time ('t') is adjusted: `t * dt_annots - 1` to align
      with 0-indexed segmentation frames.
    - The temporal search window for a match is fixed to +/- 3 frames.
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
        t *= dt_annots
        t -= 1  # because of 0 indexing
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
            t_start = max(0, t - 3)
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
            for delta in [0, 1, -1, 2, -2, 3, -3]:
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
    experiment_id = filename.split('_')[0]
    matching_row = experiments_df[experiments_df['Experiment'] == experiment_id]

    if matching_row.empty:
        return False, f"Experiment {experiment_id} not found in registry"
    
    try:
        acquisition_freq = int(matching_row['Acquisition_frequency(min)'].iloc[0])
    except (ValueError, TypeError):
        return False, f"Invalid frequency format in registry (file: {filename})"
    
    # Combined validation checks
    if acquisition_freq <= 0:
        return False, f"Invalid acquisition frequency: {acquisition_freq} (file: {filename})"
    if target_interval % acquisition_freq != 0:
        return False, f"{target_interval}min target not divisible by {acquisition_freq}min acquisition (file: {filename})"

    return True, acquisition_freq