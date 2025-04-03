import numpy as np


ATTRIBUTES = [
    'skin',
    'l_brow',
    'r_brow',
    'l_eye',
    'r_eye',
    'eye_g',
    'l_ear',
    'r_ear',
    'ear_r',
    'nose',
    'mouth',
    'u_lip',
    'l_lip',
    'neck',
    'neck_l',
    'cloth',
    'hair',
    'hat'
]

COLOR_LIST = [
    [0, 0, 0],
    [255, 85, 0],
    [255, 170, 0],
    [255, 0, 85],
    [255, 0, 170],
    [0, 255, 0],
    [85, 255, 0],
    [170, 255, 0],
    [0, 255, 85],
    [0, 255, 170],
    [0, 0, 255],
    [85, 0, 255],
    [170, 0, 255],
    [0, 85, 255],
    [0, 170, 255],
    [255, 255, 0],
    [255, 255, 85],
    [255, 255, 170],
    [255, 0, 255],
]


def vis_parsing_maps(image, segmentation_mask, save_image=False, save_path="result.png"):
    import cv2
    # Create numpy arrays for image and segmentation mask
    image = np.array(image).copy().astype(np.uint8)
    segmentation_mask = segmentation_mask.copy().astype(np.uint8)

    # Classes to include
    included_class_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17]

    # Build binary mask for selected classes
    mask = np.isin(segmentation_mask, included_class_indices).astype(np.uint8)

    # Create alpha channel: 255 where mask is 1, else 0
    alpha = (mask * 255).astype(np.uint8)

    # Create RGBA image
    rgba_image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    rgba_image[:, :, 3] = alpha

    # Crop to bounding box of non-transparent area
    y_indices, x_indices = np.where(mask == 1)
    if len(x_indices) > 0 and len(y_indices) > 0:
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        rgba_image = rgba_image[y_min:y_max, x_min:x_max]

    if save_image:
        cv2.imwrite(save_path, cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGRA))  # Ensure correct PNG format

    return rgba_image