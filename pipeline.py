import os
import sys
import logging
from datetime import datetime
import pandas as pd

try:
    from config import (
        RUN_NAME, BASE_DATA_DIR, BASE_LOG_DIR,
        EXTERNAL_PATHS, OUTPUT_DIRS, 
        SEGMENTATION_CONFIG, TRACKING_CONFIG,
        APO_MATCH_CONFIG, APO_CROP_CONFIG,
        UPSAMPLING_CONFIG
    )
except ImportError:
    print("Error: Could not import configurations from config.py.")
    print("Please ensure config.py is in the same directory and defines all required variables.")
    sys.exit(1)


# Import the necessary functions and classes from our new package
from img_proc import(
    Segmentation,
    Tracking,
    Matching,
    Cropping,
    Upsampling,
    get_image_paths,
    load_image_stack
)

# --- Configuration and Environment Setup ---

def setup_logging(log_dir, module_name):
    """Sets up file and console logging."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(log_dir, f"{module_name}_{timestamp}.log")

    log_format = "%(asctime)s | %(levelname)-8s | %(name)-17s | %(message)s"

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    return logging.getLogger("Pipeline_Main")


#def create_output_directories(base_dir, sub_dirs):
#    """Creates all required output subdirectories."""
#    for sub_dir in sub_dirs:
#        path = os.path.join(base_dir, sub_dir)
#        os.makedirs(path, exist_ok=True)



# --- Main Execution Function ---

def main(): 
    """Main function to run the image processing pipeline."""
    
    # 1. Configuration is loaded via direct import (done at the top of the file)
    
    # Use the Experiment Info path from one of the imported config dictionaries.
    # SEGMENTATION_CONFIG is a good place to pull this global path from.
    GLOBAL_EXPERIMENT_INFO_PATH = EXTERNAL_PATHS['EXPERIMENT_INFO_CSV']

    # 2. Setup Logging
    # LOG_DIR is imported directly.
    logger = setup_logging(BASE_LOG_DIR, "pipeline_main") 
    logger.info("Pipeline starting configuration and environment setup.")
    logger.info(f"Run Name: {RUN_NAME}")

    # 3. Create Output Directories
    # Use the imported module configuration dictionaries directly.
    
    output_dirs_to_create = [
        # Global Log Dir
        BASE_LOG_DIR, 
        
        # Segmentation Output Dirs
        SEGMENTATION_CONFIG['MASK_DIR'], SEGMENTATION_CONFIG['MASK_DIR_NO_FILT'],
        SEGMENTATION_CONFIG['DF_DIR'], SEGMENTATION_CONFIG['DETAILS_DIR'],
        
        # Tracking Output Dirs
        TRACKING_CONFIG['TRACKED_MASK_DIR'], TRACKING_CONFIG['TRACK_DF_DIR'], 
        TRACKING_CONFIG['PLOT_DIR'], # Using PLOT_DIR from TRACKING_CONFIG
        
        # Matching Output Dirs
        APO_MATCH_CONFIG['CSV_DIR'], APO_MATCH_CONFIG['PLOT_DIR'],
        
        # Cropping Output Dirs
        APO_CROP_CONFIG['CROPS_DIR'], APO_CROP_CONFIG['WINDOWS_DIR'], 
        APO_CROP_CONFIG['WINDOWS_DIR_20X'], APO_CROP_CONFIG['BAD_CROPS'],
        APO_CROP_CONFIG['FEATURES_DIR'], APO_CROP_CONFIG['APO_CHECK_ARRAY_DIR'],
        # Note: APO_CROP_CONFIG['PLOT_DIR'] is the same as the global PLOT_DIR/TRACKING['PLOT_DIR']
    ]
    
    # We create the directories based on the paths in the config
    for path in output_dirs_to_create:
        # path is a pathlib.Path object, os.makedirs handles it correctly
        os.makedirs(path, exist_ok=True)
        logger.info(f"Ensured directory exists: {path}")

    # 4. Load Global Data Dependencies (Experiment List)
    try:
        # Using the defined GLOBAL_EXPERIMENT_INFO_PATH
        experiments_list = pd.read_csv(GLOBAL_EXPERIMENT_INFO_PATH, header=0)
        logger.info(f"Loaded experiment list from {GLOBAL_EXPERIMENT_INFO_PATH}.")
    except FileNotFoundError:
        logger.critical(f"Critical error: experiment list not found at {GLOBAL_EXPERIMENT_INFO_PATH}. Aborting.")
        sys.exit(1)

    # 5. Get Image Paths
    img_dir_path = SEGMENTATION_CONFIG['IMG_DIR']
    image_paths = get_image_paths(img_dir_path)
    filenames = [os.path.splitext(os.path.basename(path))[0] for path in image_paths]
    logger.info(f"Found {len(filenames)} image files in {img_dir_path}.")
    
    if not image_paths:
        logger.warning("No images found. Exiting pipeline.")
        return

    # 6. Initialize Components 
    segmentation_module = Segmentation(SEGMENTATION_CONFIG)
    tracking_module = Tracking(TRACKING_CONFIG)
    matching_module = Matching(APO_MATCH_CONFIG)
    cropping_module = Cropping(APO_CROP_CONFIG)
    upsampling_module = Upsampling(UPSAMPLING_CONFIG)
    logger.info("All components initialized. Starting main loop.")
    
    # 7. Main Loop
    for path, filename in zip(image_paths, filenames):
        logger.info(f"--- Running Pipeline for {filename} ---")

        try:
            apo_file = os.path.join(
                APO_MATCH_CONFIG['APO_ANNOTATIONS_DIR'], 
                f'{filename}.csv'
            )
            # Load annotation file for the current image
            apo_annotations = pd.read_csv(
                apo_file, 
                header=None,
                names=['filename', 'x', 'y', 't'],
                on_bad_lines='skip'
            ).dropna()
            logger.info(f"Loaded {len(apo_annotations)} apoptosis annotations for {filename}.")
        except FileNotFoundError:
            logger.warning(f"Apoptosis annotation file not found for {filename} ({apo_file}). Continuing without annotations.")
            apo_annotations = pd.DataFrame(columns=['filename', 'x', 'y', 't']) # Create empty DF
        except Exception as e:
            logger.error(f"Error loading APO annotations for {filename}: {e}", exc_info=True)
            apo_annotations = pd.DataFrame(columns=['filename', 'x', 'y', 't']) # Create empty DF

        imgs = load_image_stack(path)

        
        # --- SEGMENTATION STAGE ---
        try:
            seg_out = segmentation_module.process(imgs, filename, experiments_list)
            gt_filtered, summary_df, details, gt_unfiltered = seg_out
            logger.info(f"Segmentation complete for {filename}.")
        except Exception as e:
            logger.error(f"Error in Segmentation for {filename}: {e}", exc_info=True)
            continue 

        # --- TRACKING STAGE ---
        try:
            merged_df, tracked_masks = tracking_module.process(
                filename, gt_filtered, summary_df, 
                experiments_list=experiments_list
            )
            if merged_df is not None:
                logger.info(f"Tracking successfully completed and saved outputs for {filename}.")
        except Exception as e:
            logger.error(f"Error in Tracking for {filename}: {e}", exc_info=True)
            # Decide whether to continue or abort. We will continue for now.
        
        # --- MATCHING STAGE ---
        if merged_df is not None:
            try:
                metrics, apo_annotations = matching_module.process(filename, apo_annotations,
                                        details, tracked_masks, gt_filtered,
                                        experiments_list)
                
                # Add a flag to indicate successful data generation
                matching_successful = metrics is not None
                
                if matching_successful:
                    logger.info(f"Matching successfully completed for {filename}.")
                else:
                    logger.warning(f"Matching process completed for {filename} but returned no metrics (data likely missing).")
                    
            except Exception as e:
                logger.error(f"Error in Matching for {filename}: {e}", exc_info=True)
                matching_successful = False # Ensure flag is False on error
        else:
            logger.warning(f"Skipping Matching for {filename} due to prior Tracking failure.")
            matching_successful = False # Ensure flag is False if skipped

        # --- CROPPING STAGE ---
        if matching_successful:
            try:
                cropping_module.process(filename, experiments_list,
                                        imgs, merged_df, tracked_masks,
                                        apo_annotations)
                logger.info(f"Cropping successfully completed for {filename}.")
            except Exception as e:
                logger.error(f"Error in Cropping for {filename}: {e}", exc_info=True)
        else:
            logger.warning(f"Skipping Cropping for {filename} due to prior Matching failure.")

    
    logger.info("Main loop finished. Starting finalization steps.")

    # 8. Finalize Components
    # Call finalize on each module to perform cross-file plotting/saving.
    try:
        matching_module.finalize()
        logger.info("Matching finalization complete.")
    except Exception as e:
        logger.error(f"Error during Matching finalization: {e}", exc_info=True)

    try:
        cropping_module.finalize()
        logger.info("Cropping finalization complete (Features saved, plots generated).")
    except Exception as e:
        logger.error(f"Error during Cropping finalization: {e}", exc_info=True)
        
    logger.info("Starting post-processing (Upsampling stage).")
    try:
        # Call the finalize method to process all saved crops
        upsampling_module.finalize()
        logger.info("Upsampling post-processing complete.")
    except Exception as e:
        logger.error(f"Error during Upsampling finalization: {e}", exc_info=True)


        logger.info("Pipeline execution finished.")

if __name__ == '__main__':
    main()