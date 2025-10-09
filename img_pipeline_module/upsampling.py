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
        self.parent_dir = config.get('PARENT_DIR', '')
        self.target_size = tuple(config.get('TARGET_SIZE', (128, 128)))
        self.class_mappings = config['CLASS_MAPPINGS']

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
                if len(img.shape) == 3 and img.shape[-1] == 5:
                    # Resize the image while preserving all channels
                    resized_img = resize(img, (*target_size, 5), 
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
        logger.info(f"Starting to resize saved crops to {self.target_size} for {len(self.class_mappings)} classes.")
        
        # Iterate over the configured classes
        for mapping in self.class_mappings:
            in_dir = os.path.join(self.parent_dir, mapping['IN_SUBDIR'])
            out_dir = os.path.join(self.parent_dir, mapping['OUT_SUBDIR'])
            class_name = mapping['NAME']
            
            logger.info(f"Processing class: {class_name}")
            
            # Call the worker function with the dynamic paths
            self._resize_dataset(in_dir, out_dir, self.target_size)
    
        logger.info("Upsampling complete!")