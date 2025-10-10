import logging
import os
import sys
import numpy as np
import pandas as pd
from stardist.models import StarDist2D
# We will import the helper functions from our local utils module
from utils import get_experiment_info, run_segmentation, filter_segmentation

logger = logging.getLogger(__name__)

class Segmentation:
    """
    Performs StarDist-based nuclear segmentation, object filtering, 
    and prepares masks and summary data for tracking.
    """
    
    def __init__(self, config: dict):
        """
        Initializes the Segmentation module with configuration parameters.
        
        Args:
            config: Dictionary containing segmentation-specific parameters.
        """
        self.save_data = config['SAVE_INTERMEDIATE']
        self.use_gpu = config['USE_GPU']
        self.min_nuc_size_40x = config['MIN_NUC_SIZE']
        self.min_nuc_size_20x = config['MIN_NUC_SIZE_20x']
        
        # Output directory paths (The Pipeline orchestrator will create these)
        self.mask_dir = config['MASK_DIR']
        self.mask_dir_no_filt = config['MASK_DIR_NO_FILT']
        self.df_dir = config['DF_DIR']
        self.details_dir = config['DETAILS_DIR']
        
        # StarDist Model setup
        if self.use_gpu:
            # These imports MUST be inside the class method or init
            # to be executed only when needed (if a user doesn't have a GPU)
            try:
                import gputools # Only needed if the stardist/tf backend requires it explicitly
                from csbdeep.utils.tf import limit_gpu_memory
                limit_gpu_memory(None, allow_growth=True)
            except ImportError:
                logger.warning("GPU acceleration requested but 'gputools' or 'csbdeep' dependencies not fully met.")
                self.use_gpu = False

        self.model = StarDist2D.from_pretrained("2D_versatile_fluo")
        self.axis_norm = (0, 1) # Normalization axes for 2D images
        
        logger.info("Segmentation module initialized.")

    def _determine_min_size(self, filename: str, experiments_list: pd.DataFrame) -> int:
        """
        Selects the minimum nucleus size based on image magnification.
        """
        exp_info = get_experiment_info(filename, experiments_list)
        
        if not exp_info['found'] or exp_info['magnification'] == '40x':
            min_nuc_size = self.min_nuc_size_40x
            logger.info(f"\t\tUsing 40x minimum size threshold: {min_nuc_size}")
        elif exp_info['magnification'] == '20x':
            min_nuc_size = self.min_nuc_size_20x
            logger.info(f"\t\tUsing 20x minimum size threshold: {min_nuc_size}")
        else:
            # Fallback for unexpected magnification
            min_nuc_size = self.min_nuc_size_40x
            logger.warning(f"\t\tUnexpected magnification '{exp_info['magnification']}'. Falling back to 40x size: {min_nuc_size}")
            
        return min_nuc_size

    def process(self, image_data: np.ndarray, filename: str, experiments_list: pd.DataFrame):
        """
        Runs the full segmentation process for a single image stack.

        Args:
            image_data: An array of images
            filename: The base name of the file (e.g., 'ExpXX_SiteYY').
            experiments_list: DataFrame with experiment metadata.

        Returns:
            A tuple: (gt_filtered, summary_df, details, gt_unfiltered)
            - gt_filtered: Mask stack with small objects removed.
            - summary_df: DataFrame of (obj_id, t, x, y) for filtered objects.
            - details: List of dictionaries with full StarDist results.
            - gt_unfiltered: Mask stack before size filtering.
        """
        logger.info(f"Processing {filename}: Starting Stardist segmentation.")
        
        # 1. Run Segmentation (uses helper from utils)
        # Note: run_segmentation requires the model instance
        gt_unfiltered, details = run_segmentation(image_data, self.model, self.axis_norm)
        logger.info("\tSegmentation done.")

        # 2. Determine Minimum Size
        min_nuc_size = self._determine_min_size(filename, experiments_list)

        # 3. Filter Segmentation (uses helper from utils)
        gt_filtered, summary_df = filter_segmentation(gt_unfiltered, details, min_nuc_size)
        
        # 4. Save Data (This step is handled by the Pipeline Orchestrator for clean I/O)
        if self.save_data:
            self._save_outputs(filename, gt_filtered, gt_unfiltered, summary_df, details)

        return gt_filtered, summary_df, details, gt_unfiltered

    def _save_outputs(self, filename, gt_filtered, gt_unfiltered, summary_df, details):
        """
        Private method to save segmentation outputs.
        """
        # Save masks
        mask_path = os.path.join(self.mask_dir, f'{filename}.npz')
        no_filt_path = os.path.join(self.mask_dir_no_filt, f'{filename}.npz')
        np.savez_compressed(no_filt_path, gt=gt_unfiltered)
        np.savez_compressed(mask_path, gt=gt_filtered)
        logger.info(f"\t\tFiltered Mask saved at: {mask_path}")

        # Save summary df
        df_path = os.path.join(self.df_dir, f'{filename}_pd_df.csv')
        summary_df.to_csv(df_path, index=False)
        logger.info(f"\t\tSummary-Df saved at: {df_path}")

        # Save Stardist details
        details_path = os.path.join(self.details_dir, f'{filename}.pkl')
        with open(details_path, 'wb') as f:
            # We need the pickle module here
            import pickle 
            pickle.dump(details, f)
        logger.info(f"\t\tDetails saved at: {details_path}")