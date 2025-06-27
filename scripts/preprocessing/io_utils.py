import os
import numpy as np
import pandas as pd
import logging

import tifffile as tiff
from nd2reader import ND2Reader

logger = logging.getLogger(__name__)

def load_image_stack(path):
    """
    Loads an image stack from a file based on its extension.

    Supports TIFF (.tif, .tiff) using `tifffile` and ND2 (.nd2) files using
    `ND2Reader`.

    Parameters
    ----------
    path : str
        The absolute path to the image stack file.

    Returns
    -------
    np.ndarray
        A NumPy array representing the loaded image stack. The shape will vary
        depending on the image dimensions (e.g., (T, H, W) for 2D+time).

    Raises
    ------
    ValueError
        If the file extension is not supported.
    FileNotFoundError
        If the specified file does not exist.
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
    Returns a sorted list of absolute paths for image files in a directory.

    Scans the specified directory for files with common image extensions
    (.tif, .tiff, .nd2, .npz) and returns their absolute paths, sorted
    alphabetically.

    Parameters
    ----------
    directory : str
        The path to the directory to scan for image files.

    Returns
    -------
    list of str
        A sorted list of absolute paths to the image files.

    Raises
    ------
    FileNotFoundError
        If the specified directory does not exist.
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
    Extracts specific experiment metadata from a DataFrame based on filename.

    Parses the filename to get an experiment identifier and then looks up
    corresponding details (magnification, acquisition/tracking frequencies,
    annotation frequency) from the provided `experiments_list` DataFrame.
    Returns default values if the experiment is not found.

    Parameters
    ----------
    filename : str
        The filename (e.g., "ExpXX_SiteXX") from which the experiment name
        (e.g., "ExpXX") will be extracted.
    experiments_list : pd.DataFrame
        DataFrame containing experiment parameters. Expected to have columns
        like 'Experiment', 'Magnification', 'Acquisition_frequency(min)',
        'Tracking_frequency', and 'Apo_annotation_frequency(min)'.

    Returns
    -------
    dict
        A dictionary containing experiment information. Keys include:
        'exp_name' (str): The extracted experiment name.
        'found' (bool): True if the experiment was found in `experiments_list`, False otherwise.
        'magnification' (str): Magnification (e.g., '40x', '20x'). Defaults to '40x'.
        'acquisition_freq' (float/int): Acquisition frequency in minutes.
        'tracking_freq' (float/int): Tracking frequency.
        'apo_annotation_freq' (float/int): Apoptotic annotation frequency in minutes.
        Values for frequencies will be `None` if the experiment is not found.
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