# image_pipeline_module/cropping.py

import logging
import os
import random
# import sys
import numpy as np
import pandas as pd
import tifffile as tiff
from skimage import measure
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from .utils import (
    check_temporal_compatibility, 
    crop_window,
    block_window_in_array,
    get_experiment_info
)

logger = logging.getLogger(__name__)

class Cropping:
    """
    Module for Apoptosis Window Cropping and Dataset Generation.

    Extracts time-series image crops centered on detected cells for
    machine learning applications (apoptotic, non-apoptotic, random).
    """

    def __init__(self, config: dict):
        """
        Initializes the Cropping module with necessary configuration.

        Parameters
        ----------
        config : dict
            The APO_CROP_CONFIG dictionary loaded from the main configuration.
        """
        self.config = config
        self.plot_dir = config['PLOT_DIR']
        self.run_name = config['RUN_NAME']
        
        # Directory Paths (Access via config)
        self.crops_dir = config['CROPS_DIR']
        self.windows_dir = config['WINDOWS_DIR']
        self.windows_dir_20x = config['WINDOWS_DIR_20X']
        self.bad_crops = config['BAD_CROPS']
        self.features_dir = config['FEATURES_DIR']
        self.apo_check_array_dir = config['APO_CHECK_ARRAY_DIR']


        # Lists and counters (will be updated during processing)
        self.survival_times = []
        self.all_features = []

        # Ensure output directories exist
        self._setup_directories()

        logger.info("Cropping module initialized.")

    def _setup_directories(self):
        """Creates all necessary output directories."""
        output_dirs = [
            self.crops_dir, self.windows_dir, 
            self.bad_crops, self.features_dir, 
            self.apo_check_array_dir, self.windows_dir_20x
        ]
        for path in output_dirs:
            os.makedirs(path, exist_ok=True)

        os.makedirs(os.path.join(self.features_dir, 'raw_images'), exist_ok=True)
        os.makedirs(os.path.join(self.features_dir, 'masks'), exist_ok=True)
        os.makedirs(os.path.join(self.plot_dir, self.run_name), exist_ok=True) # For feature plots

    def process(self, filename: str, experiments_list: pd.DataFrame,
                imgs: np.ndarray, merged_df: pd.DataFrame,
                tracked_masks: np.ndarray,
                apo_annotations: pd.DataFrame):
        """
        The main method to run the cropping logic for a single file.

        Parameters
        ----------
        filename : str
            The base name of the file (without extension).
        experiments_list : pd.DataFrame
            DataFrame containing experiment metadata.

        Returns
        -------
        dict
            A dictionary containing pipeline metrics (e.g., number of crops).
        """
        logger.info(f"Starting Cropping for {filename}")

        # 1. Temporal Compatibility Check
        is_valid, result = check_temporal_compatibility(
            filename, experiments_list, self.config['FRAME_INTERVAL']
        )
        if not is_valid:
            logger.warning(f"Skipping {filename} due to: {result}")
            return {'status': 'skipped', 'reason': 'temporal_incompatibility'}

        acquisition_freq = result
        step = self.config['FRAME_INTERVAL'] // acquisition_freq
        num_frames = self.config['MAX_TRACKING_DURATION'] // acquisition_freq
        num_timepoints = self.config['MAX_TRACKING_DURATION'] // self.config['FRAME_INTERVAL']

        # 2. Load DataFrames
        track_sizes = merged_df.groupby("track_id")["track_id"].transform('size')
        required_frames = (self.config['FRAME_INTERVAL'] // acquisition_freq) * (num_timepoints + 1)
        merged_df_long = merged_df[track_sizes >= required_frames].copy()
        logger.info(f"\tUsing min track length of {required_frames} frames.")

        # 3. Magnification and Window Size Setup
        exp_info = get_experiment_info(filename, experiments_list)
        magnification = exp_info.get('magnification', '40x')
        window_size = self.config['WINDOW_SIZE_20X'] if magnification == '20x' else self.config['WINDOW_SIZE']
        window_dir = self.windows_dir_20x if magnification == '20x' else self.windows_dir
        logger.info(f"\tUsing {window_size} window size for {magnification}")

        # 4. Directory Setup and Variable Initialization
        # Create file-specific output directories
        os.makedirs(os.path.join(self.crops_dir, filename), exist_ok=True)
        os.makedirs(os.path.join(self.crops_dir, f'no_apo_{filename}'), exist_ok=True)
        os.makedirs(os.path.join(self.crops_dir, f'random_{filename}'), exist_ok=True)
        os.makedirs(os.path.join(self.bad_crops, f'no_apo_{filename}'), exist_ok=True)
        os.makedirs(os.path.join(window_dir, 'apo'), exist_ok=True)
        os.makedirs(os.path.join(window_dir, 'no_apo'), exist_ok=True)
        os.makedirs(os.path.join(window_dir, 'random'), exist_ok=True)

        # Initialize tracking variables
        merged_df_long['apoptotic'] = 0
        apo_check_array = np.zeros_like(tracked_masks)

        # Initialize Crop Counter
        self.track_crop_counter = {}
        
        # --- Run the three main cropping stages ---
        
        # Store individual run metrics to return
        metrics = {}
        
        # 5. Apoptotic Cell Cropping
        apo_track_ids, apo_metrics = self._crop_apoptotic(
            filename, apo_annotations, merged_df_long, 
            apo_check_array, imgs, tracked_masks, 
            window_size, num_frames, step, acquisition_freq, window_dir
        )
        metrics.update(apo_metrics)
        num_apo_crops = apo_metrics['num_apo_crops']

        # 6. Non-Apoptotic (Healthy) Cell Cropping
        no_apo_metrics = self._crop_healthy(
            filename, merged_df_long, apo_check_array, imgs, tracked_masks, 
            window_size, num_frames, step, window_dir
        )
        metrics.update(no_apo_metrics)
        
        # 7. Random Spot Cropping
        random_metrics = self._crop_random(
            filename, num_apo_crops, apo_track_ids, 
            apo_check_array, imgs, tracked_masks, 
            window_size, num_frames, step, window_dir
        )
        metrics.update(random_metrics)


        # Save the apo_check_array for QC
        tiff.imwrite(
            os.path.join(self.apo_check_array_dir, f'{filename}.tif'),
            apo_check_array.transpose(1, 2, 0)
        )

        logger.info(f"Finished processing {filename}. Results: {metrics}")
        
        return metrics

    def finalize(self):
        """
        Runs the final steps after all files are processed (plotting, saving features).
        """
        logger.info("Starting finalization (Plotting and Feature Saving).")
        
        output_dir = os.path.join(self.plot_dir, self.run_name)
        
        if not self.all_features:
            logger.warning("No features collected for plotting/saving. Skipping finalization.")
            return

        features_df = pd.DataFrame(self.all_features)
        features_df.to_csv(os.path.join(self.features_dir, "features.csv"), index=False)
        
        # --- Plotting ---
        
        # 1. Survival Times Plot
        self._plot_survival_times(output_dir)

        # 2. Feature QC Plots
        self._plot_feature_qc(features_df, output_dir)
        
        logger.info(f"Final features saved to {self.features_dir}")
        logger.info(f"Plots saved to {output_dir}")
        return {'status': 'finalized'}


    # --- Internal Methods for Cropping Stages (Move your logic here) ---

    def _generate_crop_filename(self, filename, class_label, track_id, crop_index):
        """Generates a standardized, unique filename for a cropped sequence."""
        
        # Track ID is only relevant for apo and no_apo
        track_part = f'_T{int(track_id)}' if track_id is not None else '' 
        
        # Crop index ensures uniqueness, formatted as C01, C02, etc.
        index_part = f'_C{int(crop_index):04d}' # Use 3 digits for high capacity (C001, C999)

        # Example: 'apo_img01_T123_C001.tif' or 'random_img01_C001.tif'
        return f'{class_label}_{filename}{track_part}{index_part}.tif'

    def _find_valid_crop_sequences(
        self,
        single_cell_df, 
        step, 
        required_count,
        min_x, min_y, max_x, max_y,
        max_crops_to_extract=-1
    ):
        """
        Finds the first valid temporal offset that yields a complete sequence of 
        'required_count' frames, where every frame's (x, y) coordinates fall within 
        the spatial boundary limits (min/max_x/y).

        Returns:
        - valid_crops: List with DataFrames with the validated rows. Empty if no valid series is found.
        """

        track_id = single_cell_df['track_id'].iloc[0] if not single_cell_df.empty else "N/A"

        present_time_points = set(single_cell_df['t'].unique())
        min_t, max_t = single_cell_df['t'].min(), single_cell_df['t'].max()


        # available_track_length = len(single_cell_df)
        time_span_needed = (required_count - 1) * step

        
        valid_crops = []
        num_spatial_rejections = 0
        num_temporal_rejections = 0

        # Pre-check for minimum length
        if (max_t - min_t) < time_span_needed:
            print(f"\t\t[Track {track_id} Validation] FAILED: Time span ({max_t - min_t}) is shorter than required span ({time_span_needed}).")
            results_dict = {'track_too_short': True, 
                            'num_spatial_rejections': num_spatial_rejections,
                            'num_temporal_rejections': num_temporal_rejections
                        }
            return [], results_dict

        # extract all valid t0 so that we do not run into an indexing error
        possible_t0_list = sorted([
            t for t in present_time_points if (t + time_span_needed) <= max_t
        ])

        for start_t in possible_t0_list:
            required_time_points = list(range(start_t, start_t + time_span_needed + 1, step))

            is_temporally_consistent = all(t in present_time_points for t in required_time_points)

            if not is_temporally_consistent:
                # print(f"\t\t[Track {track_id} Validation] Start Time {start_t} FAILED: Temporal consistency violated (missing frame(s) in the sequence).")
                num_temporal_rejections += 1
                continue

            # Extract rows based on 't'
            potential_rows_df = single_cell_df[single_cell_df['t'].isin(required_time_points)]
            
            if len(potential_rows_df) != required_count:
                # This indicates a severe bug in the time-point mapping if it triggers
                print(f"CRITICAL: Length mismatch after consistency check for t={start_t}")
                continue


            # Check if all x-coordinates are within the horizontal boundary
            x_valid = (
                (potential_rows_df['x'] >= min_x) & 
                (potential_rows_df['x'] <= max_x)
            ).all()
            
            # Check if all y-coordinates are within the vertical boundary
            y_valid = (
                (potential_rows_df['y'] >= min_y) & 
                (potential_rows_df['y'] <= max_y)
            ).all()

            if x_valid and y_valid:
                # Found a valid temporal and spatial alignment!
                valid_crops.append(potential_rows_df.copy())
                if max_crops_to_extract != -1 and len(valid_crops) >= max_crops_to_extract:
                    results_dict = {'track_too_short': False, 
                            'num_spatial_rejections': num_spatial_rejections,
                            'num_temporal_rejections': num_temporal_rejections
                        }
                    return valid_crops, results_dict
            else:
                num_spatial_rejections += 1
                # print(f"\t\t[Track {track_id} Validation] start time {start_t} FAILED: Spatial boundaries violated for at least one frame.")
                
        # If the loop completes without finding a valid sequence
        if not valid_crops:
            num_starts_checked = len(possible_t0_list)
            print(
                f"\t\t[Track {track_id} Validation] FAILED: Exhausted all {num_starts_checked} "
                f"start indices (Spatial Rej: {num_spatial_rejections}, Temporal Rej: {num_temporal_rejections})."
            )
            # print(f"\t\t[Track {track_id} Validation] FAILED: Exhausted all {num_starts_checked} possible start indices; no valid series found.")
        
        results_dict = {'track_too_short': False, 
                        'num_spatial_rejections': num_spatial_rejections,
                        'num_temporal_rejections': num_temporal_rejections
                    }
        return valid_crops, results_dict

    def _sample_valid_crops(
        self, 
        valid_crops_list, 
        max_crops_limit, 
        prioritize_first=False
    ):
        """
        Samples a subset of crop sequences from the list based on a limit and strategy.

        Args:
            valid_crops_list (list): List of DataFrames (the valid crop sequences).
            max_crops_limit (int): The maximum number of crops to return. Use -1 to take all.
            prioritize_first (bool): If True, index 0 is guaranteed to be included
                                    (used for APO tracks). If False, sampling is purely random.

        Returns:
            list: A subset of sampled DataFrames.
        """
        list_length = len(valid_crops_list)
        
        if list_length == 0:
            return []

        # 1. Determine the actual number of samples to take
        # If limit is -1 or greater than list length, take all available.
        if max_crops_limit == -1 or max_crops_limit > list_length:
            n_samples = list_length
        else:
            n_samples = max_crops_limit

        # Handle the case where the limit is 0 (though less likely)
        if n_samples == 0:
            return []

        # 2. Determine the sampling indices based on strategy
        if prioritize_first:
            # Strategy A: APO - Always include index 0, sample N-1 from the rest.
            
            # If n_samples is 1, random.sample will correctly return an empty list.
            # If n_samples > 1, sample n_samples - 1 from the remaining indices.
            other_indices = random.sample(range(1, list_length), n_samples - 1)
            indices = [0] + other_indices
        else:
            # Strategy B: Healthy - Purely random sample of N indices from the whole list.
            indices = random.sample(range(list_length), n_samples)
        
        # Sort the indices for consistent processing order
        indices.sort()
        
        # 3. Create and return the sampled list
        return [valid_crops_list[i] for i in indices]


    def _crop_apoptotic(self, filename, apo_annotations, merged_df_long, apo_check_array, 
                        imgs, tracked_masks, window_size, num_frames, 
                        step, acquisition_freq, window_dir):
        """Logic for cropping apoptotic cells."""
        # --- Metrics Setup ---
        num_apo_crops = 0
        num_no_match = 0
        num_wrong_size = 0
        num_track_too_short = 0
        num_skipped_tracks = 0
        successful_track_ids = set()

        # --- Configuration Access ---
        NUM_BLOCKED_FRAMES = self.config['NUM_BLOCKED_FRAMES']
        CROPS_PER_TRACK_APO = self.config['CROPS_PER_TRACK_APO']
        MIN_REQUIRED_LENGTH = (num_frames // step) + 1

        apo_track_ids = pd.DataFrame(columns=['track_id', 'apo_start_t'])


        # Calculate safe zone to extract crops from
        h_img, w_img = imgs[0].shape[:2]
        margin = window_size // 2

        min_x = margin
        min_y = margin
        max_x = w_img - margin - 1
        max_y = h_img - margin - 1


        logger.info("\tStarting cropping for apo cells")

        # --- Main Loop (Processing Annotations) ---
        for i, row in tqdm(apo_annotations.iterrows(),
                           total=len(apo_annotations),
                           desc="Processing Annotations"):
            current_track_id = row.loc['matching_track']
            current_t = row.loc['correct_t'] + row.loc['delta_ts']

            apo_track_ids.loc[i, 'track_id'] = current_track_id
            apo_track_ids.loc[i, 'apo_start_t'] = current_t

            if not np.isscalar(current_track_id):
                current_track_id = current_track_id.iloc[0]

            # 1. Handle No Match (Track ID == 0) and Block
            if current_track_id == 0:
                annot_x = int(row['x'])
                annot_y = int(row['y'])
                annot_t = int(row['correct_t'])
                window_size_no_match = 2 * window_size
                num_block_no_match = 2 * self.config['NUM_BLOCKED_FRAMES']

                block_window_in_array(
                    apo_check_array, annot_t, annot_x, annot_y,
                    window_size_no_match, num_block_no_match,
                    acquisition_freq
                )
                logger.debug("\t\tSkipping annotation, no match found.")
                num_no_match += 1
                continue

            # 2. Extract Single Track Data
            is_correct_track = merged_df_long['track_id'] == current_track_id
            is_valid_time = merged_df_long['t'] >= current_t
            single_cell_df = merged_df_long.loc[is_correct_track & is_valid_time].copy()
            
            if single_cell_df.empty:
                logger.warning(f"Track: {current_track_id} not found in csv.")
                continue

            # Update Global State (mark cells as apo in df + update stats)
            merged_df_long.loc[is_correct_track & is_valid_time, 'apoptotic'] = 1
            num_entries = single_cell_df.shape[0]
            self.survival_times.append(num_entries)
            
            
            # Block window in apo_check_array
            last_row = single_cell_df.iloc[-1]
            last_x = int(last_row['x'])
            last_y = int(last_row['y'])
            last_t = int(last_row['t'])

            block_window_in_array(
                apo_check_array, last_t, last_x, last_y,
                window_size, NUM_BLOCKED_FRAMES,
                acquisition_freq
            )

            # 3. Find Valid Sequences
            valid_crops_list, status_dict = self._find_valid_crop_sequences(
                single_cell_df,
                step,
                MIN_REQUIRED_LENGTH,
                min_x, min_y, max_x, max_y,
                max_crops_to_extract=-1 # Crucial: Find ALL sequences
            )

            num_wrong_size += status_dict['num_spatial_rejections']

            if not valid_crops_list:
                logger.debug(f"\t\tSkipping track: {current_track_id}. Reason: No valid crops found (Track too short/spatial bounds failed).")
                num_skipped_tracks += 1
                if status_dict['track_too_short']:
                    num_track_too_short += 1
                continue

            # 4. Sample Crops (Prioritize the sequence starting closest to APO event)
            sampled_crops_list = self._sample_valid_crops(
                valid_crops_list, 
                CROPS_PER_TRACK_APO, 
                prioritize_first=True
            )

            if not sampled_crops_list:
                continue

            successful_track_ids.add(current_track_id)

            # 5. Extract and Save Sampled Crops
            for crop_idx_in_track, positions_to_crop_df in enumerate(sampled_crops_list):
                windows = []
                
                # Extract and check window dimensions (size check is redundant if _find... is perfect)
                for _, sc_row in positions_to_crop_df.iterrows():
                    window = crop_window(
                        imgs[int(sc_row['t'])],
                        int(sc_row['x']),
                        int(sc_row['y']),
                        window_size
                    )
                    windows.append(window)

                # Assert final sequence length and size (Redundant but safe debug check)
                assert len(windows) == MIN_REQUIRED_LENGTH and all(w.shape == (window_size, window_size) for w in windows), \
                    "CRITICAL ERROR: Sequence validation failed after sampling!"
                
                sub_windows = np.asarray(windows)
                current_track_id = int(current_track_id)

                # --- File Naming and Saving ---
                # Update counter for unique filename generation
                counter_key = (filename, current_track_id, 'apo')
                current_idx = self.track_crop_counter.get(counter_key, 0) + 1
                self.track_crop_counter[counter_key] = current_idx

                final_name = self._generate_crop_filename(
                    filename=filename,
                    class_label='apo',
                    track_id=current_track_id,
                    crop_index=current_idx
                )

                target_path = os.path.join(window_dir, 'apo', final_name)
                tiff.imwrite(target_path, sub_windows.transpose(1, 2, 0))

                # --- Softlink Creation ---
                link_name_for_qc = f'trackID_{current_track_id}_crop_{current_idx}.tif'
                link_path = os.path.join(self.crops_dir, filename, link_name_for_qc)
                
                try:
                    if os.path.exists(link_path) or os.path.islink(link_path):
                        os.remove(link_path)
                    os.symlink(target_path, link_path)
                except Exception as e:
                    logger.error(f"Failed to create soft link for {link_name_for_qc}. Error: {e}")

                num_apo_crops += 1
                
        # --- Logging and Return ---
        num_successful_tracks = len(successful_track_ids)
        logger.info(f"\t\tValid crops of apo cells found for {num_apo_crops} crops from {num_successful_tracks}/{len(apo_annotations)} tracks.")
        logger.info(f"\t\tNum annotations with no match: {num_no_match}")
        logger.info(f"\t\tNum tracks too short: {num_track_too_short}")
        logger.info(f"\t\tNum tracks skipped (spatial/temporal): {num_skipped_tracks}")

        return apo_track_ids, {
            'num_apo_crops': num_apo_crops,
            'apo_tracks_successful': num_successful_tracks,
            'apo_no_match': num_no_match,
            'apo_wrong_size': num_wrong_size,
            'apo_track_too_short': num_track_too_short,
            'apo_tracks_skipped': num_skipped_tracks
        }
        

    def _crop_healthy(self, filename, merged_df_long, apo_check_array, imgs, tracked_masks, 
                      window_size, num_frames, step, window_dir):
        """
        Logic for cropping non-apoptotic (healthy) cells using a
        pre-validation and sampling strategy.
        
        1. Finds all spatially valid crop windows for each track.
        2. Samples a configured number of crops from these valid windows.
        3. Performs a final check for blocking by nearby apoptotic cells.
        4. Calculates features, performs QC, and saves valid crops.
        """
        


        logger.info('\tStarting cropping for non-apo cells.')
    
        # --- 1. Configuration Access ---
        try:
            CROPS_PER_TRACK_HEALTHY = self.config['CROPS_PER_TRACK_HEALTHY']
            
            # QC parameters
            ECCENTRICITY_THR = self.config['ECCENTRICITY_THR']
            CROP_STD_THR = self.config['CROP_STD_THR']
            CROP_MEAN_INT_THR = self.config['CROP_MEAN_INT_THR']
            SOLIDITY_THR = self.config['SOLIDITY_THR']
        except KeyError as e:
            logger.error(f"Missing required config parameter for healthy cropping: {e}")
            raise

        # Calculate MIN_REQUIRED_LENGTH exactly as done in _crop_apoptotic
        MIN_REQUIRED_LENGTH = (num_frames // step) + 1
        
        # --- 2. Calculate Safe Zone ---
        # Replicates the exact logic from _crop_apoptotic
        h_img, w_img = imgs[0].shape[:2]
        margin = window_size // 2

        min_x = margin
        min_y = margin
        max_x = w_img - margin - 1
        max_y = h_img - margin - 1
        
        expected_shape = (window_size, window_size)

        # --- 3. Data Preparation & Metrics Setup ---
        long_no_apo_df = merged_df_long[merged_df_long['apoptotic'] == 0]
        unique_track_ids = np.unique(long_no_apo_df['track_id'])

        num_healthy_crops = 0
        num_blocked = 0         # Blocked by nearby apo
        num_wrong_size = 0      # Spatial rejections from _find_valid_crop_seq..
        num_track_too_short = 0 # Temporal rejections from _find_valid_crop..
        num_filtered = 0        # Failed post-crop QC
        num_skipped_tracks = 0  # Tracks with no valid windows at all

        # --- 4. Main Loop (Iterating over Tracks) ---
        for i, track_id in tqdm(enumerate(unique_track_ids),
                                    total=len(unique_track_ids),
                                    desc="Cropping non-apo Windows"):
            track_id = int(track_id)
            single_cell_df = long_no_apo_df.loc[
                long_no_apo_df['track_id'] == track_id
            ].copy()

            # --- 4a. Find Valid Sequences ---
            valid_crops_list, status_dict = self._find_valid_crop_sequences(
                single_cell_df,
                step,
                MIN_REQUIRED_LENGTH,
                min_x, min_y, max_x, max_y,
                max_crops_to_extract=-1 # Find all valid sequences
            )
            num_wrong_size += status_dict.get('num_spatial_rejections', 0)

            if not valid_crops_list:
                num_skipped_tracks += 1
                if status_dict.get('track_too_short', False):
                    num_track_too_short += 1
                continue # No valid crops to process for this track

            # --- 4b. Sample Crops ---
            # (prioritize_first=False for healthy cells)
            sampled_crops_list = self._sample_valid_crops(
                valid_crops_list, 
                CROPS_PER_TRACK_HEALTHY,
                prioritize_first=False
            )
        
            if not sampled_crops_list:
                continue

            # --- 4c. Extract and Save Sampled Crops ---
            for positions_to_crop_df in sampled_crops_list:
                windows = []
                mask_windows = []
                is_blocked_crop = False
                crop_start_t = positions_to_crop_df['t'].min()

                for _, sc_row in positions_to_crop_df.iterrows():
                    current_t, current_x, current_y = int(sc_row['t']), int(sc_row['x']), int(sc_row['y'])
                    
                    # --- Blocking Check ---
                    apo_window = crop_window(apo_check_array[current_t], current_x, current_y, window_size)
                    if np.any(apo_window == 1):
                        logger.debug(f"\t\tTrack {track_id}, crop at t={crop_start_t} blocked by nearby apoptosis.")
                        num_blocked += 1
                        is_blocked_crop = True
                        break # Exit inner frame loop for this crop

                    # --- Crop (Spatial safety is pre-validated) ---
                    window = crop_window(imgs[current_t], current_x, current_y, window_size)
                    mask_window = crop_window(tracked_masks[current_t], current_x, current_y, window_size)
            
                    mask_window = mask_window.astype(int)
                    mask_window[mask_window != track_id] = 0
                    
                    windows.append(window)
                    mask_windows.append(mask_window)

                if is_blocked_crop:
                    continue # Move to the next sampled crop

                # --- Final Validation (Sanity Check) ---
                enough_frames = len(windows) == MIN_REQUIRED_LENGTH
                all_frames_correct_size = all(w.shape == expected_shape for w in windows)
                
                if not enough_frames or not all_frames_correct_size:
                    logger.warning(f"CRITICAL: Post-validation failed for non-apo track {track_id}!")
                    continue

                # --- 5d. Feature Calculation ---
                current_features = []
                for mask, img in zip(mask_windows, windows):
                    try:
                        props = measure.regionprops_table(mask, img, properties=['label', 'eccentricity',
                                                                                    'intensity_mean', 'intensity_std',
                                                                                    'solidity'])
                        feature_df = pd.DataFrame(props)
                        if not feature_df.empty:
                            current_features.append(feature_df)
                    except Exception as e:
                        logger.warning(f"Feature extraction failed for track {track_id} at t={crop_start_t}. Error: {e}")
                
                if not current_features:
                    logger.debug(f"No features found for track {track_id} at t={crop_start_t} (e.g., empty mask). Skipping.")
                    continue

                track_features = pd.concat(current_features, ignore_index=True)
                track_features['x'] = single_cell_df['x'].iloc[0]
                track_features['y'] = single_cell_df['y'].iloc[0]
                track_features['t'] = crop_start_t
                mean_features = track_features.mean()
                
                windows = np.asarray(windows)

                # --- 5e. File Naming and Saving ---
                counter_key = (filename, track_id, 'no_apo')
                current_idx = self.track_crop_counter.get(counter_key, 0) + 1
                self.track_crop_counter[counter_key] = current_idx

                final_name = self._generate_crop_filename(
                    filename=filename,
                    class_label='no_apo',
                    track_id=track_id,
                    crop_index=current_idx
                )

                mean_features['filename'] = final_name
                self.all_features.append(mean_features)

                # TODO: Save feature data? Maybe not necessary
                tiff.imwrite(os.path.join(self.features_dir, 'raw_images', final_name), windows)
                tiff.imwrite(os.path.join(self.features_dir, 'masks', final_name), np.asarray(mask_windows))

                # --- 5f. QC Check (Unique to healthy logic) ---
                # TODO: Make optional
                mean_eccentricity = mean_features['eccentricity']
                mean_intensity = mean_features['intensity_mean']
                mean_std = mean_features['intensity_std']
                mean_solidity = mean_features['solidity']

                is_filtered = any((mean_eccentricity < ECCENTRICITY_THR,
                                    mean_std > CROP_STD_THR,
                                    mean_intensity > CROP_MEAN_INT_THR,
                                    mean_solidity < SOLIDITY_THR))

                if is_filtered:
                    tiff.imwrite(os.path.join(self.bad_crops, f'no_apo_{filename}', final_name), windows.transpose(1, 2, 0))
                    num_filtered += 1
                else:
                    # --- Save Good Crop ---
                    target_path = os.path.join(window_dir, 'no_apo', final_name)
                    tiff.imwrite(target_path, windows.transpose(1, 2, 0))

                    # --- 5g. Create Symlink ---
                    link_name_for_qc = f'trackID_{track_id}_crop_{current_idx}.tif' 
                    link_path = os.path.join(self.crops_dir, f'no_apo_{filename}', link_name_for_qc)

                    try:
                        if os.path.exists(link_path) or os.path.islink(link_path):
                            os.remove(link_path)
                        os.symlink(os.path.abspath(target_path), link_path)
                    except Exception as e:
                        logger.error(f"Failed to create soft link {link_path}. Error: {e}")
                    
                    num_healthy_crops += 1
        
        # --- 6. Logging and Return ---
        logger.info(f"\t\tFound {num_healthy_crops} valid crops of healthy cells.")
        total_rejected = num_blocked + num_wrong_size + num_track_too_short + num_filtered + num_skipped_tracks
        if total_rejected > 0:
            logger.info(f"\t\t\t{total_rejected} potential crops were rejected in total.")
            logger.info(f"\t\t\t{num_blocked} blocked by nearby apoptosis.")
            logger.info(f"\t\t\t{num_wrong_size} rejected for spatial/border issues.")
            logger.info(f"\t\t\t{num_track_too_short} tracks too short.")
            logger.info(f"\t\t\t{num_filtered} filtered out in post-crop QC.")
            logger.info(f"\t\t\t{num_skipped_tracks} tracks had no valid windows at all.")
            
        return {
            'num_healthy_crops': num_healthy_crops,
            'healthy_rejected': total_rejected,
            'healthy_blocked': num_blocked,
            'healthy_wrong_size': num_wrong_size,
            'healthy_track_too_short': num_track_too_short,
            'healthy_filtered_qc': num_filtered,
            'healthy_tracks_skipped': num_skipped_tracks
        }


    def _crop_random(self, filename, num_apo_crops, apo_track_ids, apo_check_array, 
                     imgs, tracked_masks, window_size, num_frames, 
                     step, window_dir):
        """Logic for cropping random spots."""
        num_random_tracks = num_apo_crops # Try to match the number of apo crops
        logger.info(f'\tStarting cropping for {num_random_tracks} random spots.')

        if len(imgs) == 0:
            return {'num_random_crops': 0}

        img_height, img_width = imgs[0].shape
        iter_count = 0
        crop_count = 0
        
        # Max 2000 iterations to prevent infinite loop
        while (crop_count < num_random_tracks) and (iter_count <= 2000):
            
            # Select a random start frame within a valid range for a full sequence
            start_t = np.random.randint(0, len(imgs) - num_frames)

            # Randomly generate valid (x, y) coordinates for the crop center
            half_window = window_size // 2
            random_x = np.random.randint(half_window, img_width - half_window)
            random_y = np.random.randint(half_window, img_height - half_window)

            windows = []
            is_valid_sequence = True
            
            # Extract a window for each frame in the track duration, checking validity
            for t in range(start_t, start_t + num_frames + 1, step):
                window = crop_window(imgs[t], random_x, random_y, window_size)
                track_id_mask_crop = crop_window(tracked_masks[t], random_x, random_y, window_size)
                apo_check_array_crop = crop_window(apo_check_array[t], random_x, random_y, window_size)
                present_track_ids = track_id_mask_crop.flatten().tolist()
                
                # Filter out apoptotic tracks that were active at or before time t
                current_apo_track_ids = apo_track_ids.loc[apo_track_ids['apo_start_t'] <= t]
                current_apo_ids = set(current_apo_track_ids['track_id'])

                is_pixel_not_in_apo = not any(pixel in current_apo_ids for pixel in present_track_ids)
                is_window_correct_size = window.shape == (window_size, window_size)
                is_area_blocked = np.any(apo_check_array_crop == 1)
                
                if (
                    is_pixel_not_in_apo 
                    and is_window_correct_size 
                    and not is_area_blocked
                ):
                    windows.append(window)
                else:
                    is_valid_sequence = False # Sequence is invalid if any frame fails
                    if not is_pixel_not_in_apo:
                        logger.debug("\t\tCurrent window contains an apoptotic cell.")
                    elif not is_window_correct_size:
                        logger.debug("\t\tWindow shape does not match target size (boundary issue).")
                    elif is_area_blocked:
                        logger.debug("\t\tArea is in a blocked region (too close to an apo event).")
                    break # Stop processing this sequence
            
            
            required_len = (self.config['MAX_TRACKING_DURATION'] // self.config['FRAME_INTERVAL']) + 1
            if is_valid_sequence and len(windows) == required_len:
                windows = np.asarray(windows)

                final_name = self._generate_crop_filename(
                    filename=filename,
                    class_label='random',
                    track_id=None,
                    crop_index=crop_count + 1 # Use the existing crop_count
                )    

                target_path = os.path.join(window_dir, 'random', final_name)
                tiff.imwrite(target_path, windows.transpose(1, 2, 0))

                link_name_for_qc = f'ID_{crop_count}.tif' 
                
                link_path = os.path.join(self.crops_dir, f'random_{filename}', link_name_for_qc)

                # 3. Create the soft link in the CROPS_DIR pointing to the original file
                try:
                    if os.path.exists(link_path) or os.path.islink(link_path):
                        os.remove(link_path)
                        
                    # os.symlink(source, link_name)
                    os.symlink(target_path, link_path)
                    
                except Exception as e:
                    logger.error(f"Failed to create soft link for random crop {link_name_for_qc}. Error: {e}")

                crop_count += 1
            
            iter_count += 1
            
        logger.info(f"\t\tFinished random cropping with {crop_count} crops after {iter_count} iterations.")

        return {
            'num_random_crops': crop_count,
            'random_iterations': iter_count
        }
        # --- End of your Random Cropping Logic Refactored ---


    # --- Internal Methods for Plotting ---
    def _plot_survival_times(self, output_dir):
        """Generates and saves the survival times histogram."""
        if not self.survival_times:
            logger.warning("No survival times data to plot.")
            return

        plt.figure(figsize=(10, 6))
        plt.hist(self.survival_times, bins=20, range=(0, 200), edgecolor='black', alpha=0.7)
        plt.xlim(0, 200)
        plt.xlabel("Survival Time (frames)", fontsize=12)
        plt.ylabel("Number of Cells", fontsize=12)
        plt.title("Histogram of Cell Survival Times (Frames)", fontsize=14)
        plt.grid(True, alpha=0.4)
        plot_filename = os.path.join(output_dir, "survival_times_histogram.png")
        plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
        plt.close()

    def _plot_feature_qc(self, features_df, output_dir):
        """Generates and saves the feature QC plots."""
        # The complex plotting function from your code
        def plot_feature(data, feature_name, cutoff, xlim, comparison='>', units='', bins=20, output_dir='.'):
            # ... (Your plot_feature function logic goes here) ...
            
            # --- Start of your plot_feature function Refactored ---
            text_xpos = 0.95
            text_ha = 'right'
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Calculate histogram
            counts, bins, patches = ax.hist(data, bins=bins, edgecolor='black', alpha=0.7)
            
            # Add KDE plot
            if len(data) > 1:
                kde = gaussian_kde(data)
                x_vals = np.linspace(bins[0], bins[-1], 1000) # Use the range of the bins
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
            # --- End of your plot_feature function Refactored ---
            
        # Call plot_feature for each QC metric
        plot_feature(features_df['eccentricity'], 'eccentricity', self.config['ECCENTRICITY_THR'], xlim=(0,1), comparison='<', output_dir=output_dir)
        plot_feature(features_df['intensity_std'], 'intensity_std', self.config['CROP_STD_THR'], xlim=(0,3000), comparison='>', output_dir=output_dir)
        plot_feature(features_df['solidity'], 'solidity', self.config['SOLIDITY_THR'], xlim=(0.8,1), comparison='<', output_dir=output_dir)
        plot_feature(features_df['intensity_mean'], 'intensity_mean', self.config['CROP_MEAN_INT_THR'], xlim=(0, 5500), comparison='>', output_dir=output_dir)