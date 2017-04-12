"""File with some general helper functions, e.g. image manipulation."""
from __future__ import print_function, division
import numpy as np
import math
import matplotlib.pyplot as plt
import imgaug as ia

def to_aspect_ratio_add(image, target_ratio, pad_mode="constant", pad_cval=0, return_paddings=False):
    """Resize an image to a desired aspect ratio by adding pixels to it
    (usually black ones, i.e. zero-padding)."""
    height = image.shape[0]
    width = image.shape[1]
    ratio = width / height

    pad_top = 0
    pad_bottom = 0
    pad_left = 0
    pad_right = 0

    if ratio < target_ratio:
        # vertical image, height > width
        diff = (target_ratio * height) - width
        pad_left = int(math.ceil(diff / 2))
        pad_right = int(math.floor(diff / 2))
    elif ratio > target_ratio:
        # horizontal image, width > height
        diff = ((1/target_ratio) * width) - height
        pad_top = int(math.ceil(diff / 2))
        pad_bottom = int(math.floor(diff / 2))

    if any([val > 0 for val in [pad_top, pad_bottom, pad_left, pad_right]]):
        # constant_values creates error if pad_mode is not "constant"
        if pad_mode == "constant":
            image = np.pad(image, ((pad_top, pad_bottom), \
                                   (pad_left, pad_right), \
                                   (0, 0)), \
                                  mode=pad_mode, constant_values=pad_cval)
        else:
            image = np.pad(image, ((pad_top, pad_bottom), \
                                   (pad_left, pad_right), \
                                   (0, 0)), \
                                  mode=pad_mode)

    result_ratio = image.shape[1] / image.shape[0]
    assert target_ratio - 0.1 < result_ratio < target_ratio + 0.1, \
        "Wrong result ratio: " + str(result_ratio)

    if return_paddings:
        return image, (pad_top, pad_right, pad_bottom, pad_left)
    else:
        return image

def draw_heatmap(img, heatmap, alpha=0.5):
    """Draw a heatmap overlay over an image."""
    assert len(heatmap.shape) == 2 or \
        (len(heatmap.shape) == 3 and heatmap.shape[2] == 1)
    assert img.dtype in [np.uint8, np.int32, np.int64]
    assert heatmap.dtype in [np.float32, np.float64]

    if img.shape[0:2] != heatmap.shape[0:2]:
        heatmap_rs = np.clip(heatmap * 255, 0, 255).astype(np.uint8)
        heatmap_rs = ia.imresize_single_image(
            heatmap_rs[..., np.newaxis],
            img.shape[0:2],
            interpolation="nearest"
        )
        heatmap = np.squeeze(heatmap_rs) / 255.0

    cmap = plt.get_cmap('jet')
    heatmap_cmapped = cmap(heatmap)
    heatmap_cmapped = np.delete(heatmap_cmapped, 3, 2)
    heatmap_cmapped = heatmap_cmapped * 255
    mix = (1-alpha) * img + alpha * heatmap_cmapped
    mix = np.clip(mix, 0, 255).astype(np.uint8)
    return mix

def imresize_sidelen(image, maxval, pick_func=min, interpolation=None, force_even_sidelens=False):
    """Resize an image so that one of its size is not larger than a maximum
    value."""
    height, width = image.shape[0], image.shape[1]
    currval = pick_func(height, width)
    if currval < maxval:
        if force_even_sidelens:
            newheight = height
            newwidth = width
            if newheight % 2 != 0:
                newheight += 1
            if newwidth % 2 != 0:
                newwidth += 1
            if newheight == height and newwidth == width:
                return np.copy(image)
            else:
                return ia.imresize_single_image(
                    image,
                    (newheight, newwidth),
                    interpolation=interpolation
                )
        else:
            return np.copy(image)
    else:
        scale_factor = maxval / currval
        newheight, newwidth = int(height * scale_factor), int(width * scale_factor)
        if force_even_sidelens:
            if newheight % 2 != 0:
                newheight += 1
            if newwidth % 2 != 0:
                newwidth += 1
        return ia.imresize_single_image(
            image,
            (newheight, newwidth),
            interpolation=interpolation
        )
