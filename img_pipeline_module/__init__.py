# img_pipeline_module/__init__.py

# Import core components
from .segmentation import Segmentation
from .tracking import Tracking
from .matching import Matching
from .cropping import Cropping
from .upsampling import Upsampling

# Import essential utilities to be accessible at the package level
from .utils import (
    load_image_stack,
    get_image_paths,
    get_experiment_info,
    run_segmentation,
    filter_segmentation,
    get_btrack_params,
    remove_outlier_frames,
    run_tracking,
    convert_obj_to_track_ids,
    plot_track_lengths,
    fill_track_gaps_vectorized,
    check_temporal_compatibility,
    match_annotations,
    plot_matching_distances,
    crop_window,
    block_window_in_array,
)

# Define __all__ to list the public objects imported when a user does 'from img_pipeline_module import *'
__all__ = [
    "Segmentation",
    "Tracking",
    "Matching",
    "Cropping",
    "Upsampling",
    # Utilities
    "load_image_stack",
    "get_image_paths",
    "get_experiment_info",
    "run_segmentation",
    "filter_segmentation",
    "get_btrack_params",
    "remove_outlier_frames",
    "run_tracking",
    "convert_obj_to_track_ids",
    "plot_track_lengths",
    "fill_track_gaps_vectorized",
    "check_temporal_compatibility",
    "match_annotations",
    "plot_matching_distances",
    "crop_window",
    "block_window_in_array",
]