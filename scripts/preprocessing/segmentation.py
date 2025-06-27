from tqdm import tqdm
import numpy as np
import pandas as pd
from csbdeep.utils import normalize
from skimage.morphology import remove_small_objects
import logging



from .io_utils import load_image_stack

logger = logging.getLogger(__name__)

def run_segmentation(path, model, axis_norm):
    """
    Performs instance segmentation on an image stack.

    Loads an image stack, normalizes each image, and then applies a provided
    segmentation model to predict instance masks and their details.

    Parameters
    ----------
    path : str
        Path to the image stack file (e.g., a TIFF stack).
    model
        A segmentation model object (e.g., StarDist model) with a
        `predict_instances` method.
    axis_norm : tuple or None
        Axes along which to normalize the images (e.g., (0,1) for 2D,
        None for global). Passed directly to the `normalize` function.

    Returns
    -------
    gt : np.ndarray
        A 3D NumPy array of segmented instance masks (time, height, width),
        where each non-zero pixel represents an object ID.
    details : list of dict
        A list of dictionaries, one per frame, containing prediction details
        from the segmentation model (e.g., centroids, probabilities).
    """
    h2b_imgs = load_image_stack(path)
    h2b_imgs_normal = np.asarray([normalize(img, 1, 99.8, axis=axis_norm) for img in h2b_imgs])
    
    gt = []
    details = []
    for x in tqdm(h2b_imgs_normal, desc="Segmenting"):
        labels, det = model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False)
        gt.append(labels)
        details.append(det)
    gt = np.asarray(gt)
    return gt, details

def filter_segmentation(gt, details, min_size):
    """
    Filters segmented objects by size and creates a summary DataFrame.

    Removes objects from segmentation masks that are smaller than a specified
    minimum size. For the remaining objects, it extracts their unique IDs,
    time points, and centroid coordinates, compiling this information into
    a pandas DataFrame.

    Parameters
    ----------
    gt : np.ndarray
        A 3D NumPy array of segmented instance masks (time, height, width).
        Each non-zero pixel represents an object ID.
    details : list of dict
        A list of dictionaries, one per frame, containing prediction details
        from the segmentation model. Expected to have a 'points' key
        with object centroid coordinates.
    min_size : int
        The minimum pixel area for an object to be retained. Objects with
        fewer pixels than this size will be removed.

    Returns
    -------
    gt_filtered : np.ndarray
        A copy of the input `gt` array with objects smaller than `min_size`
        removed (i.e., their pixels set to zero). Data type is `np.uint16`.
    summary_df : pd.DataFrame
        A DataFrame summarizing the filtered objects with columns:
        'obj_id' (int): Original unique ID of the object.
        't' (int): Time frame index.
        'x' (float): X-coordinate (column index) of the object's centroid.
        'y' (float): Y-coordinate (row index) of the object's centroid.
    """
    logger.info(f"\tRemoving objects smaller than {min_size}.")
    num_frames = gt.shape[0]
    gt_filtered = np.zeros_like(gt, dtype=np.uint16)
    df_list = []
    for frame in range(num_frames):
        gt_filtered[frame] = remove_small_objects(gt[frame], min_size=min_size)
        unique_ids = np.unique(gt_filtered[frame])
        unique_ids = unique_ids[unique_ids > 0]
        x, y = [], []
        timepoint = np.full_like(unique_ids, frame)
        current_details = details[frame]['points']
        for obj_id in unique_ids:
            position = current_details[obj_id - 1]  # Adjust indexing as needed
            x.append(position[1])
            y.append(position[0])
        current_df = pd.DataFrame({'obj_id': unique_ids, 't': timepoint, 'x': x, 'y': y})
        df_list.append(current_df)
    summary_df = pd.concat(df_list, ignore_index=True)
    logging.info("\t\tDone!")
    return gt_filtered, summary_df