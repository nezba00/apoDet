import os
import numpy as np
from skimage.transform import resize
import tifffile
from pathlib import Path
import glob
from tqdm import tqdm  # For progress bar

PARENT_DIR = '/home/nbahou/myimaging/apoDet/data/dataset2'
IN_PATH_1 = PARENT_DIR + '/windows_20x_2cat/apo'
IN_PATH_2 = PARENT_DIR + '/windows_20x_2cat/no_apo'
OUT_PATH_1 = PARENT_DIR + '/windows_20x_2cat_resize_128/apo'
OUT_PATH_2 = PARENT_DIR + '/windows_20x_2cat_resize_128/no_apo'

TARGET_SIZE = (128, 128)


def resize_image_dataset(input_dir, output_dir, target_size=(128, 128)):
    """
    Resize all .tif images in input_dir and save them to output_dir with the same names.
    
    Args:
        input_dir: Directory containing original .tif images
        output_dir: Directory to save resized images
        target_size: Tuple (height, width) for the target size
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all .tif files
    image_files = glob.glob(os.path.join(input_dir, "*.tif"))
    
    print(f"Found {len(image_files)} .tif images in {input_dir}")
    
    # Process each image with a progress bar
    for image_path in tqdm(image_files, desc="Resizing images"):
        try:
            # Get image filename
            filename = os.path.basename(image_path)
            
            # Load the .tif image
            img = tifffile.imread(image_path)
            
            # Check if the image has the expected dimensions (32x32x5)
            if len(img.shape) == 3 and img.shape[2] == 5:
                # Resize the image while preserving all 5 channels
                resized_img = resize(img, (*target_size, 5), 
                                    anti_aliasing=True, 
                                    preserve_range=True)
                
                # Save the resized image as .tif
                output_path = os.path.join(output_dir, filename)
                
                # Convert back to the same dtype as the original
                resized_img = resized_img.astype(img.dtype)
                #print(resized_img.shape)
                # Save with tifffile which handles multi-channel tiffs well
                tifffile.imwrite(output_path, resized_img)
            else:
                print(f"WARNING: Skipping {filename} - unexpected shape: {img.shape}")
        
        except Exception as e:
            print(f"ERROR: Processing {image_path}: {e}")

# Process both directories
def main():
    # Replace these with your actual paths
    input_dir1 = IN_PATH_1
    output_dir1 = OUT_PATH_1
    
    input_dir2 = IN_PATH_2
    output_dir2 = OUT_PATH_2
    
    # Target size - using 128x128
    target_size = TARGET_SIZE
    
    print(f"Starting to resize images to {target_size}...")
    
    # Process first directory
    resize_image_dataset(input_dir1, output_dir1, target_size)
    
    # Process second directory
    resize_image_dataset(input_dir2, output_dir2, target_size)
    
    print("Resizing complete!")


if __name__ == "__main__":
    main()