import logging
import os
import numpy as np
import pandas as pd
# import sys # Needed for np.load

# Import all tracking and utility helpers
from .utils import (
    remove_outlier_frames, 
    run_tracking, 
    convert_obj_to_track_ids,
    get_btrack_params,
    plot_track_lengths,
    fill_track_gaps_vectorized
)

logger = logging.getLogger(__name__)

class Tracking:
    """
    The Tracking component handles loading segmentation results, applying
    Bayesian Tracking (BTrack), and generating tracked masks and dataframes.
    """
    
    def __init__(self, config: dict):
        """
        Initializes the Tracking module with configuration parameters.
        
        Args:
            config: Dictionary containing tracking-specific parameters.
        """
        self.config = config # Store the full config for dynamic lookup
        
        # Input/Output paths
        self.tracked_mask_dir = config['TRACKED_MASK_DIR'] # Output
        self.track_df_dir = config['TRACK_DF_DIR']       # Output
        self.plot_dir = config['PLOT_DIR']               # Output
        
        # Tracking settings
        self.run_name = config['RUN_NAME']
        self.min_track_len = config['TRK_MIN_LEN']
        
        # Configure Btrack logging
        logging.getLogger('btrack').setLevel(logging.WARNING)
        
        logger.info("Tracking module initialized.")

    def process(self, filename: str,
                mask: np.ndarray, 
                segmentation_df: pd.DataFrame, 
                experiments_list: pd.DataFrame):
        """
        Runs the full tracking process for a single image stack.

        Args:
            mask: An Array with segmentation masks
            segmentation_df: DataFrame with centroids and obj_id
            experiments_list: DataFrame with experiment metadata.

        Returns:
            pd.DataFrame: The final merged DataFrame with track IDs.
            np.ndarray: The mask stack with track IDs.
        """
        logger.info(f"\tStarting Tracking for {filename}.")


        # -- 1. Pre-processing: Remove Outlier Frames --
        mask_filt, outlier_indices = remove_outlier_frames(mask)
        logger.info(f'\t\t{len(outlier_indices)} outlier frames replaced with zeros.')

        # -- 2. Config Selection --
        bt_config_path, track_radius = get_btrack_params(
            filename, experiments_list, self.config
        )
        logger.info(f"\t\tBTrack Config: {os.path.basename(bt_config_path)}, Radius: {track_radius}px")

        # -- 3. Run Tracking --
        dfBTracks = run_tracking(mask_filt, bt_config_path, track_radius)
        
        # -- 4. Merge Data --
        merged_df = segmentation_df.merge(
            dfBTracks.drop(columns=["x", "y"]), # Drop BTrack's x,y, keep Stardist's
            on=["obj_id", "t"],
            how="left"
        )
        logger.info("\t\tMerged information from BTrack and Stardist.")

        merged_df = fill_track_gaps_vectorized(merged_df.copy())

        # -- 5. Convert Masks --
        tracked_masks = convert_obj_to_track_ids(mask_filt, merged_df)
        logger.info("\t\tConverted object IDs to track IDs in masks.")

        # -- 6. Save Outputs (I/O Handler) --
        self._save_outputs(filename, merged_df, tracked_masks)
        
        # -- 7. Plot Track Lengths --
        plot_track_lengths(
            merged_df, 
            self.min_track_len, 
            filename, 
            self.plot_dir, 
            self.run_name
        )

        return merged_df, tracked_masks

    def _save_outputs(self, filename, merged_df, tracked_masks):
        """Private method to save tracking outputs to disk."""
        
        # Save merged DataFrame
        merge_df_path = os.path.join(self.track_df_dir, f"{filename}.csv")
        merged_df.to_csv(merge_df_path, index=False)
        logger.info(f"\t\tSaved merged tracking DataFrame at: {merge_df_path}")

        # Save tracked masks
        mask_path = os.path.join(self.tracked_mask_dir, f'{filename}.npz')
        np.savez_compressed(mask_path, gt=tracked_masks)
        logger.info(f"\t\tSaved tracked masks at: {mask_path}")