import pandas as pd
from tqdm import tqdm
import numpy as np







def match_annotations(apo_annotations, details, tracked_masks, gt_filtered):
    """
    Match manual annotations with stardist detections.
    
    Parameters:
        apo_annotations (DataFrame): DataFrame containing manual annotations.
        details (dict or list): Details loaded from the stardist output, e.g., centroids.
        tracked_masks (ndarray): The mask array with btrack track IDs.
        gt_filtered (ndarray): The filtered segmentation array.
        
    Returns:
        DataFrame: The apo_annotations DataFrame updated with matching information.
        dict: A dictionary containing evaluation metrics (e.g. num_matches, num_mismatches,
              distances for alternative matching, etc.)
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
                counts = np.bincount(previous_frames)
                match_id = np.argmax(counts)
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
    """Check compatibility between acquisition frequency and target interval.
    
    Returns:
        tuple: (is_valid: bool, result: str|int)
        - If valid: (True, acquisition_freq: int)
        - If invalid: (False, error_message: str)
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