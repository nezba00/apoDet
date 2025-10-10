import logging
import os
import pickle
import numpy as np
import pandas as pd

# Import utilities
from utils import (
    get_experiment_info, # Needed to get acquisition frequency
    match_annotations,
    plot_matching_distances
)

logger = logging.getLogger(__name__)

class Matching:
    """
    The Matching component handles matching manual annotations with automated
    segmentation/tracking results and performs evaluation/plotting.
    """
    
    def __init__(self, config: dict):
        """
        Initializes the Matching module with configuration parameters.
        """
        self.config = config
        
        # Output paths
        self.csv_dir = config['CSV_DIR']          # Matched annotations output
        self.plot_dir = config['PLOT_DIR']        # Plots output
        self.run_name = config['RUN_NAME']
        
        # Global metrics accumulator (for cross-file plotting)
        self.all_metrics = []
        self.total_matches = 0
        self.total_mismatches = 0
        
        logger.info("Matching module initialized.")

    def process(self, filename: str,
                apo_annotations: pd.DataFrame, details:pd.DataFrame,
                tracked_masks: np.ndarray, gt_filtered: np.ndarray,
                experiments_list: pd.DataFrame):
        """
        Runs the matching process for a single file.

        Args:
            filename: The base name of the file (e.g., 'ExpXX_SiteYY').
            apo_annotations: Dataframe with cols
            experiments_list: DataFrame with experiment metadata.
            
        Returns:
            dict: The metrics dictionary for this file, or None on failure.
        """
        logger.info(f"\tStarting Matching for {filename}.")

        # --- 1. Determine Time Multiplier ---
        exp_info = get_experiment_info(filename, experiments_list)
        # Use default apo annotation frequency of 5 min if not specified
        apo_annotation_freq = exp_info['apo_annotation_freq'] if exp_info['apo_annotation_freq'] is not None else 5 
        
        if not exp_info['found'] or exp_info['acquisition_freq'] is None:
            # Assume 5 min apo annots and 5 min acq if info is missing or invalid (multiplier = 1)
            multiplier = 1 
            logger.warning(f"\t\tExperiment info missing/invalid, assuming multiplier=1 for {filename}.")
        else:
            dt_acq = int(exp_info['acquisition_freq'])
            dt_apo_annots = int(apo_annotation_freq)
            multiplier = dt_apo_annots // dt_acq
            logger.info(f"\t\tTime multiplier: {dt_apo_annots} / {dt_acq} = {multiplier}.")

        # --- 2. Run Matching ---
        apo_annotations_match, metrics = match_annotations(
            apo_annotations, details, tracked_masks, gt_filtered, multiplier
        )

        # --- 3. Update Metrics and Save ---
        self.total_matches += metrics['num_matches']
        self.total_mismatches += metrics['num_mismatches']
        self.all_metrics.append(metrics)
        
        logger.info(f"\t\tFound {metrics['num_matches']} matches and {metrics['num_mismatches']} mismatches.")
        success_rate = (metrics['num_matches']*100)/(metrics['num_matches']+metrics['num_mismatches']) if (metrics['num_matches']+metrics['num_mismatches']) > 0 else 0
        logger.info(f"\t\t{success_rate:.2f}% Success Rate")

        # Apply multiplier to time column for later analysis
        apo_annotations_match['correct_t'] = apo_annotations_match['t'] * multiplier

        # Save output
        output_path = os.path.join(self.csv_dir, f'{filename}.csv')
        apo_annotations_match.to_csv(output_path, index=False)
        logger.info(f"\t\tApo-Annotations with new centroids saved at: {output_path}")

        return metrics, apo_annotations_match

    def finalize(self):
        """Called once at the end of the pipeline to perform final plotting."""
        logger.info("Finalizing Matching results: Generating distance plots.")
        
        # Plotting uses the accumulated metrics from all processed files
        plot_matching_distances(self.all_metrics, self.plot_dir, self.run_name)
        
        total_attempts = self.total_matches + self.total_mismatches
        final_success = (self.total_matches*100)/total_attempts if total_attempts > 0 else 0
        
        logger.info(f"Matching finished. Total matches: {self.total_matches}, Total mismatches: {self.total_mismatches}.")
        logger.info(f"Final Success Rate: {final_success:.2f}%")