from tqdm import tqdm
import numpy as np
import pandas as pd
from csbdeep.utils import normalize
from skimage.morphology import remove_small_objects
import logging



from .io_utils import load_image_stack



def run_segmentation(path, model, axis_norm):
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
    logging.info("\tRemoving objects smaller than {MIN_NUC_SIZE}.")
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