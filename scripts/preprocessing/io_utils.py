import os
import numpy as np
import pandas as pd
import logging

import tifffile as tiff
from nd2reader import ND2Reader

logger = logging.getLogger(__name__)

def load_image_stack(path):
    """
    Load an image stack from a file based on its extension.
    
    Uses tifffile for TIFF files and ND2Reader for ND2 files.
    """
    if path.endswith(('.tif', '.tiff')):
        # Load TIFF file using tifffile
        return tiff.imread(path)
    elif path.endswith('.nd2'):
        # Load ND2 file using ND2Reader and convert it to a numpy array
        with ND2Reader(path) as nd2:
            return np.array(nd2)
    else:
        raise ValueError(f"Unsupported file format for file: {path}")

def get_image_paths(directory):
    """
    Returns a list of absolute paths of all TIFF and ND2 files in a directory.
    """
    valid_extensions = ('.tif', '.tiff', '.nd2', '.npz')
    paths = [
        os.path.abspath(os.path.join(directory, f))
        for f in os.listdir(directory)
        if f.endswith(valid_extensions)
    ]
    return sorted(paths)

def get_experiment_info(filename, experiments_list):
    """
    Extracts experiment information based on the filename from the experiments list.
    
    Parameters
    ----------
    filename : str
        The filename to extract experiment name from
    experiments_list : pandas.DataFrame
        DataFrame containing experiment information
    logger : logging.Logger
        Logger instance for logging messages
        
    Returns
    -------
    dict
        Dictionary containing experiment information including:
        - magnification: The magnification used ('40x' or '20x')
        - acquisition_freq: Acquisition frequency in minutes
        - tracking_freq: Tracking frequency
        - apo_annotation_freq: Apoptotic annotation frequency in minutes
        - exp_name: Name of the experiment
        - found: Boolean indicating if the experiment was found in the list
    """
    # Extract experiment name from filename
    exp_name = filename.split('_')[0]
    
    # Initialize the result dictionary with default values
    result = {
        'exp_name': exp_name,
        'found': False,
        'magnification': '40x',  # Default value
        'acquisition_freq': None,
        'tracking_freq': None,
        'apo_annotation_freq': None
    }
    
    # Find the experiment in the list
    exp_row = experiments_list[experiments_list['Experiment'] == exp_name]
    
    if exp_row.empty:
        logger.info(f"\t\tExperiment {exp_name} not found in experiment info.")
    else:
        # Update result with values from the experiment list
        result['found'] = True
        result['magnification'] = exp_row['Magnification'].values[0]
        result['acquisition_freq'] = exp_row['Acquisition_frequency(min)'].values[0]
        result['tracking_freq'] = exp_row['Tracking_frequency'].values[0]
        result['apo_annotation_freq'] = exp_row['Apo_annotation_frequency(min)'].values[0]
        
        logger.info(f"\t\tFound experiment {exp_name} in experiment info.")
    
    return result