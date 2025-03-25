def crop_window(img, center_x, center_y, window_size):
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