import numpy as np

def crop_window(img, center_x, center_y, window_size):
    """
    Crops a square window from a 2D image around a specified center.

    The window's boundaries are clamped to the image dimensions, so it will
    not extend beyond the image edges.

    Parameters
    ----------
    img : np.ndarray
        The 2D image array to crop from.
    center_x : int
        The X-coordinate (column index) of the desired window center.
    center_y : int
        The Y-coordinate (row index) of the desired window center.
    window_size : int
        The desired side length of the square window in pixels.

    Returns
    -------
    np.ndarray
        The cropped 2D image window.
    """
    # Check if number is even, add one if so
    # if window_size%2 == 0:
    #     window_size += 1
    half_window_size = window_size // 2
    x_from = max(center_x - half_window_size, 0)
    x_to = min(center_x + half_window_size, img.shape[1])
    y_from = max(center_y - half_window_size, 0)
    y_to = min(center_y + half_window_size, img.shape[0])
    window = img[y_from:y_to, x_from:x_to]

    return window

def block_window_in_array(array, t, x, y, window_size, num_blocked_frames, acquisition_freq):
    """
    Blocks a window in the given array around the specified coordinates.
    
    Args:
        array: The 3D array to block in (t, y, x format)
        t: Time index
        x: X coordinate
        y: Y coordinate
        window_size: Size of the window
        num_blocked_frames: Number of frames to block forward in time
        acquisition_freq: Acquisition frequency for scaling time blocks
    """
    max_t, max_y, max_x = array.shape
    half_window = window_size // 2
    
    # Spatial indices (clamped to array bounds)
    x_start = max(0, x - half_window)
    x_end = min(max_x, x_start + window_size)
    y_start = max(0, y - half_window)
    y_end = min(max_y, y_start + window_size)
    
    # Temporal indices (clamped)
    t_end = min(max_t, t + num_blocked_frames // acquisition_freq)
    
    # Set the block to 1 if there's a valid time range
    if t_end > t:
        array[t:t_end, y_start:y_end, x_start:x_end] = 1