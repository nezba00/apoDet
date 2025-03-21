import logging
import pandas as pd
import numpy as np


import btrack
from btrack.constants import BayesianUpdates

def run_tracking(gt_filtered, fovX, fovY, bt_config_file, track_radius):
    logging.info("\tStarting tracking")
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
    logging.info("\t\tTracking Done.")
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
            tracked_masks[t][mask_frame == obj_id] = track_id

    return tracked_masks