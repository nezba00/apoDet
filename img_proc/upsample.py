import os
import numpy as np
import pandas as pd
from skimage.transform import resize
import tifffile
import glob
import logging
from tqdm import tqdm 

logger = logging.getLogger(__name__)

class Upsampling:
    """Handles post-processing operations like resizing/upsampling saved crops."""
    def __init__(self, config):
        self.config = config
        self.parent_dir = config['PARENT_DIR']
        self.scratch_dir = config['SCRATCH_DIR']
        self.target_size = tuple(config.get('TARGET_SIZE', (128, 128)))
        self.class_mappings = config['CLASS_MAPPINGS']

        self.input_base_dir_name = config['INPUT_WINDOW_DIR_BASE']
        self.output_base_dir_name = config['OUTPUT_WINDOW_DIR_BASE']

    def _resize_dataset(self, input_dir, output_dir, target_size):
        """Internal worker function to resize all images in a directory."""
        os.makedirs(output_dir, exist_ok=True)
        image_files = glob.glob(os.path.join(input_dir, "*.tif"))
        
        logger.info(f"Found {len(image_files)} .tif images in {input_dir}")
        
        for image_path in tqdm(image_files, desc=f"Resizing {os.path.basename(input_dir)}"):
            try:
                filename = os.path.basename(image_path)
                img = tifffile.imread(image_path)
                
                # Assumes channels are last (H, W, C)
                if len(img.shape) == 3:
                    # Resize the image while preserving all channels
                    num_channels = img.shape[-1]
                    resized_img = resize(img, (*target_size, num_channels), 
                                         anti_aliasing=True, 
                                         preserve_range=True)
                    
                    output_path = os.path.join(output_dir, filename)
                    resized_img = resized_img.astype(img.dtype)
                    tifffile.imwrite(output_path, resized_img)
                else:
                    logger.warning(f"Skipping {filename} - unexpected shape: {img.shape}")
            
            except Exception as e:
                logger.error(f"Processing {image_path}: {e}", exc_info=True)


    def finalize(self):
        """
        Processes all crops across all files. It iterates over class mappings, 
        constructs the full paths, and calls the resize worker function.
        """
        logger.info(f"Starting to resize saved crops to {self.target_size} for {len(self.class_mappings)} classes.")
        
        # 1. Define the full, root paths for I/O based on PARENT_DIR and BASE_NAME
        # self.parent_dir is the full data path (Path object)
        input_base_path = os.path.join(self.scratch_dir, self.input_base_dir_name)
        output_base_path = os.path.join(self.parent_dir, self.output_base_dir_name)
        
        # Ensure the overall output base directory exists
        os.makedirs(output_base_path, exist_ok=True)

        # 2. Iterate over the configured classes
        for mapping in self.class_mappings:
            class_subdir = mapping['SUBDIR'] # e.g., 'apo' or 'no_apo'
            class_name = mapping['NAME']
            
            # 3. Construct the full, specific input and output directory paths
            # Path: PARENT_DIR / INPUT_BASE_NAME / CLASS_SUBDIR
            in_dir = os.path.join(input_base_path, class_subdir)
            
            # Path: PARENT_DIR / OUTPUT_BASE_NAME / CLASS_SUBDIR
            out_dir = os.path.join(output_base_path, class_subdir)

            # Check if input directory exists before trying to resize
            if not os.path.isdir(in_dir):
                logger.warning(f"Input directory not found for class '{class_name}': {in_dir}. Skipping.")
                continue

            logger.info(f"Processing class: {class_name} ({in_dir} -> {out_dir})")
            
            # 4. Call the worker function
            # (Requires your original _resize_dataset implementation)
            self._resize_dataset(in_dir, out_dir, self.target_size)
    
        logger.info("Upsampling complete!")