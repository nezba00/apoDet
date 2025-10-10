# image_pipeline_module/cropping.py

import logging
import os
# import sys
import numpy as np
import pandas as pd
import tifffile as tiff
from skimage import measure
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from utils import (
    check_temporal_compatibility, 
    crop_window,
    get_image_paths, 
    load_image_stack, 
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
        target_size = window_size
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
        
        # --- Run the three main cropping stages ---
        
        # Store individual run metrics to return
        metrics = {}
        
        # 5. Apoptotic Cell Cropping
        apo_track_ids, apo_metrics = self._crop_apoptotic(
            filename, apo_annotations, merged_df_long, 
            apo_check_array, imgs, tracked_masks, 
            window_size, target_size, num_frames, step, acquisition_freq, window_dir
        )
        metrics.update(apo_metrics)
        num_apo_crops = apo_metrics['num_apo_crops']

        # 6. Non-Apoptotic (Healthy) Cell Cropping
        no_apo_metrics = self._crop_healthy(
            filename, merged_df_long, apo_check_array, imgs, tracked_masks, 
            window_size, target_size, num_frames, step, window_dir
        )
        metrics.update(no_apo_metrics)
        
        # 7. Random Spot Cropping
        random_metrics = self._crop_random(
            filename, num_apo_crops, apo_track_ids, 
            apo_check_array, imgs, tracked_masks, 
            window_size, target_size, num_frames, step, window_dir
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

    def _crop_apoptotic(self, filename, apo_annotations, merged_df_long, apo_check_array, 
                        imgs, tracked_masks, window_size, target_size, num_frames, 
                        step, acquisition_freq, window_dir):
        """Logic for cropping apoptotic cells."""
        # This is where the first section of your main loop logic goes.
        # It needs to return a dictionary of metrics and the apo_track_ids DataFrame.
        # ... (Your apo cropping logic goes here, refactored to use 'self' for parameters) ...
        # NOTE: For brevity, I am omitting the body of this huge function, 
        # but you should copy/paste your existing logic into this method, 
        # replacing global variables and configurations with 'self.variable' 
        # and passing necessary arguments.

        # Example replacement:
        # Before: 
        # if len(single_cell_df) < num_frames + 1:
        #     logger.debug(...)
        # After:
        # if len(single_cell_df) < num_frames + 1:
        #     logger.debug(...)
        
        # --- Start of your APO Cropping Logic Refactored ---
        num_apo_crops = 0
        num_no_match = 0
        num_wrong_size = 0
        num_track_too_short = 0

        apo_track_ids = pd.DataFrame(columns=['track_id', 'apo_start_t'])

        logger.info("\tStarting cropping for apo cells")
        for i, row in tqdm(apo_annotations.iterrows(),
                           total=len(apo_annotations),
                           desc="Processing Annotations"):
            current_track_id = row.loc['matching_track']
            current_t = row.loc['correct_t'] + row.loc['delta_ts']

            apo_track_ids.loc[i, 'track_id'] = current_track_id
            apo_track_ids.loc[i, 'apo_start_t'] = current_t

            if not np.isscalar(current_track_id):
                current_track_id = current_track_id.iloc[0]

            if current_track_id == 0:
                annot_x = int(row['x'])
                annot_y = int(row['y'])
                annot_t = int(row['correct_t'])
                window_size_no_match = 2 * window_size
                num_block_no_match = 2 * self.config['NUM_BLOCKED_FRAMES']

                block_window_in_array(
                    apo_check_array,
                    annot_t,
                    annot_x,
                    annot_y,
                    window_size_no_match,
                    num_block_no_match,
                    acquisition_freq
                )
                logger.debug("\t\tSkipping annotation, no match found.")
                num_no_match += 1
                continue

            is_correct_track = merged_df_long['track_id'] == current_track_id
            is_valid_time = merged_df_long['t'] >= current_t
            single_cell_df = merged_df_long.loc[is_correct_track & is_valid_time]
            
            if single_cell_df.empty:
                logger.warning(f"Track: {current_track_id} not found in csv.")
                continue

            # Count track length after manual apoptosis annotation
            num_entries = single_cell_df.shape[0]
            self.survival_times.append(num_entries)

            # Mark cells as apoptotic
            merged_df_long.loc[is_correct_track & is_valid_time, 'apoptotic'] = 1
            
            # Block window in apo_check_array
            last_row = single_cell_df.iloc[-1]
            last_x = int(last_row['x'])
            last_y = int(last_row['y'])
            last_t = int(last_row['t'])

            block_window_in_array(
                apo_check_array,
                last_t,
                last_x,
                last_y,
                window_size,
                self.config['NUM_BLOCKED_FRAMES'],
                acquisition_freq
            )    

            if len(single_cell_df) < num_frames + 1:
                logger.debug(f"\t\tSkipping track: {current_track_id}. Track lost too quickly. len = {len(single_cell_df)}")
                num_track_too_short += 1
                continue

            upper_t_limit = current_t + num_frames + step
            single_cell_df = single_cell_df.loc[single_cell_df['t'] < upper_t_limit]

            windows = []
            for _, sc_row in single_cell_df.iterrows():
                window = crop_window(imgs[int(sc_row['t'])],
                                     int(sc_row['x']),
                                     int(sc_row['y']),
                                     window_size)
                if window.shape == (target_size, target_size):
                    windows.append(window)
                else:
                    windows.append(None)

            chosen_offset = None
            for offset in range(step):
                sub_windows = windows[offset::step]
                all_frames_valid = all(x is not None for x in sub_windows)
                enough_frames = (len(sub_windows) == (num_frames // step) + 1)
                if all_frames_valid and enough_frames:
                    chosen_offset = offset
                    break

            if chosen_offset is None:
                logger.debug("\t\tAt least one of the windows does not have the correct size or sequence.")
                num_wrong_size += 1
            else:
                sub_windows = windows[chosen_offset::step]
                sub_windows = np.asarray(sub_windows)
                if len(sub_windows) == (num_frames // step) + 1:
                    # Save to CROPS_DIR for QC
                    tiff.imwrite(os.path.join(self.crops_dir, filename, f'trackID_{current_track_id}.tif'), 
                                 sub_windows.transpose(1, 2, 0))
                    # Save to WINDOW_DIR for ML
                    tiff.imwrite(os.path.join(window_dir, 'apo', f'apo_{filename}_{i}.tif'), 
                                 sub_windows.transpose(1, 2, 0))
                    num_apo_crops += 1
                else:
                    logger.warning(f'\t\tWrong size after temporal sampling. Length = {len(windows)}.')
        
        logger.info(f"\t\tValid crops of apo cells found for {num_apo_crops}/{len(apo_annotations)}")
        logger.info(f"\t\tNum annotations with no match: {num_no_match}")
        logger.info(f"\t\tNum wrong size after cropping: {num_wrong_size}")
        logger.info(f"\t\tNum tracks too short: {num_track_too_short}")
        
        return apo_track_ids, {
            'num_apo_crops': num_apo_crops,
            'apo_no_match': num_no_match,
            'apo_wrong_size': num_wrong_size,
            'apo_track_too_short': num_track_too_short
        }
        # --- End of your APO Cropping Logic Refactored ---


    def _crop_healthy(self, filename, merged_df_long, apo_check_array, imgs, tracked_masks, 
                      window_size, target_size, num_frames, step, window_dir):
        """Logic for cropping non-apoptotic (healthy) cells."""
        # This is where the second section of your main loop logic goes.
        # ... (Your healthy cropping logic goes here) ...
        
        # --- Start of your Healthy Cropping Logic Refactored ---
        logger.info('\tStarting cropping for non-apo cells.')
        long_no_apo_df = merged_df_long[merged_df_long['apoptotic'] == 0]
        unique_track_ids = np.unique(long_no_apo_df['track_id'])

        num_healthy_crops = 0
        rejected_windows = 0
        num_blocked = 0
        num_wrong_size = 0
        num_track_too_short = 0
        num_filtered = 0

        for i, track_id in tqdm(enumerate(unique_track_ids),
                                total=len(unique_track_ids),
                                desc="Cropping non-apo Windows"):
            track_id = int(track_id)
            single_cell_df = long_no_apo_df.loc[
                long_no_apo_df['track_id'] == track_id
                ]
            start_t = min(single_cell_df['t'])
            single_cell_df = single_cell_df.loc[
                single_cell_df['t'] <= start_t + num_frames
                ]
            
            if len(single_cell_df) < (num_frames + 1):
                logger.debug("\t\tSkipping current object, track too short.")
                num_track_too_short += 1
                rejected_windows += 1
                continue

            windows = []
            mask_windows = []
            break_loop = False
            for _, sc_row in single_cell_df.iterrows():
                window = crop_window(imgs[int(sc_row['t'])], int(sc_row['x']), int(sc_row['y']), window_size)
                mask_window = crop_window(tracked_masks[int(sc_row['t'])], int(sc_row['x']), int(sc_row['y']), window_size)
                apo_window = crop_window(apo_check_array[int(sc_row['t'])], int(sc_row['x']), int(sc_row['y']), window_size)
                
                mask_window = mask_window.astype(int)
                mask_window[mask_window != track_id] = 0
                
                is_blocked = np.any(apo_window == 1)
                is_window_valid = (window.shape == (target_size, target_size)) and (not is_blocked)

                if is_window_valid:
                    windows.append(window)
                    mask_windows.append(mask_window)
                elif is_blocked:
                    logger.debug("\t\tCrop blocked because of apoptotic event closeby")
                    num_blocked += 1
                    rejected_windows += 1
                    break_loop = True
                    break
                else:
                    num_wrong_size += 1
                    rejected_windows += 1
                    logger.debug(f"\t\tRejected window at (t={int(sc_row['t'])}): Blocked={is_blocked}, Shape={window.shape}")
                    break_loop = True
                    break
            
            if break_loop or len(windows) < num_frames + 1:
                continue

            # Feature calculation and QC (Only run on full sequences)
            if len(windows) == (num_frames) + 1:
                current_features = []
                for mask, img in zip(mask_windows, windows):
                    props = measure.regionprops_table(mask, img, properties=['label', 'eccentricity',
                                                                            'intensity_mean', 'intensity_std',
                                                                            'solidity', ])
                    feature_df = pd.DataFrame(props)
                    if not feature_df.empty:
                        current_features.append(feature_df)
                        
                if current_features:
                    track_features = pd.concat(current_features, ignore_index=True)
                    # Add static info for mean calculation later
                    track_features['x'] = single_cell_df['x'].iloc[0]
                    track_features['y'] = single_cell_df['y'].iloc[0]
                    track_features['t'] = start_t
                    mean_features = track_features.mean()
                    mean_features['filename'] = f'cell_{filename}_{track_id}.tif'
                    self.all_features.append(mean_features)

                    mean_eccentricity = mean_features['eccentricity']
                    mean_intensity = mean_features['intensity_mean']
                    mean_std = mean_features['intensity_std']
                    mean_solidity = mean_features['solidity']

                    windows = np.asarray(windows)
                    
                    # Save all crops for features analysis
                    tiff.imwrite(os.path.join(self.features_dir, 'raw_images', f'cell_{filename}_{track_id}.tif'), windows)
                    tiff.imwrite(os.path.join(self.features_dir, 'masks', f'cell_{filename}_{track_id}.tif'), np.asarray(mask_windows))

                    # QC Check
                    is_filtered = any((mean_eccentricity < self.config['ECCENTRICITY_THR'],
                                       mean_std > self.config['CROP_STD_THR'],
                                       mean_intensity > self.config['CROP_MEAN_INT_THR'],
                                       mean_solidity < self.config['SOLIDITY_THR']))
                    
                    if is_filtered:
                        tiff.imwrite(os.path.join(self.bad_crops, f'no_apo_{filename}', f'trackID_{track_id}.tif'), windows.transpose(1, 2, 0))
                        rejected_windows += 1
                        num_filtered += 1
                    else:
                        # Save to CROPS_DIR for QC
                        tiff.imwrite(os.path.join(self.crops_dir, f'no_apo_{filename}', f'trackID_{track_id}.tif'), windows[::step].transpose(1, 2, 0))
                        # Save to WINDOW_DIR for ML
                        tiff.imwrite(os.path.join(window_dir, 'no_apo', f'no_apo_{filename}_{i}.tif'), windows[::step].transpose(1, 2, 0))
                        num_healthy_crops += 1
        
        logger.info(f"\t\tFound {num_healthy_crops} valid crops of healthy cells.")
        if rejected_windows > 0:
            logger.info(f"\t\t\t{rejected_windows} crops were rejected in total.")
            logger.info(f"\t\t\t{num_blocked} blocked.")
            logger.info(f"\t\t\t{num_wrong_size} wrong size.")
            logger.info(f"\t\t\t{num_track_too_short} tracks too short.")
            logger.info(f"\t\t\t{num_filtered} filtered out in qc.")

        return {
            'num_healthy_crops': num_healthy_crops,
            'healthy_rejected': rejected_windows,
            'healthy_blocked': num_blocked,
            'healthy_wrong_size': num_wrong_size,
            'healthy_track_too_short': num_track_too_short,
            'healthy_filtered_qc': num_filtered
        }
        # --- End of your Healthy Cropping Logic Refactored ---
        
        
    def _crop_random(self, filename, num_apo_crops, apo_track_ids, apo_check_array, 
                     imgs, tracked_masks, window_size, target_size, num_frames, 
                     step, window_dir):
        """Logic for cropping random spots."""
        # This is where the third section of your main loop logic goes.
        # ... (Your random cropping logic goes here) ...

        # --- Start of your Random Cropping Logic Refactored ---
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
                is_window_correct_size = window.shape == (target_size, target_size)
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
                # Save to CROPS_DIR for QC
                tiff.imwrite(os.path.join(self.crops_dir, f'random_{filename}', f'ID_{crop_count}.tif'), 
                             windows.transpose(1, 2, 0))
                # Save to WINDOW_DIR for ML
                tiff.imwrite(os.path.join(window_dir, 'random', f'random_{filename}_{crop_count}.tif'), 
                             windows.transpose(1, 2, 0))
                crop_count += 1
            
            iter_count += 1
            
        logger.info(f"\t\tFinished random cropping with {crop_count} crops after {iter_count} iterations.")

        return {
            'num_random_crops': crop_count,
            'random_iterations': iter_count
        }
        # --- End of your Random Cropping Logic Refactored ---

    # --- Internal Methods for Plotting (Extracted from your final block) ---
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