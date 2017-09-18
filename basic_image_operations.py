import numpy as np
import scipy
from skimage import io, filters

def load_image(path):
    return scipy.misc.imread(path)


def identify_image(im):
    """
    Decides if the image is a screen shot of the Cures screen
    or the Ingredients screen.
    im is a numpy array of 3 dimensions, encoding the screen shot
    """
    score_cures = np.mean(im[1025:1065, 1130:1180, 0])
    score_ingredients = np.mean(im[1025:1065, 675:720, 0])
    if score_cures < 177.5:
        return 'cures'
    if score_ingredients < 177.5:
        return 'ingredients'
    else:
        return 'other'


def cut_main_screen(im):
    """
    Selects the main screen from the screen shots of the Cures and
    the Ingredients screens.
    """
    top = 0
    left = 0
    bottom = 980
    right = 1350
    return im[top:bottom, left:right].copy()


def make_bw(im, th=150):
    """
    Transforms the image into black-and-white,
    encoded by -1 and +1 values, respectively.
    """
    im_gray = np.mean(im, axis=2)
    im_binary = im_gray > th
    boolean_to_numbers = lambda b: 1 if b else -1
    v_boolean_to_numbers = np.vectorize(boolean_to_numbers)
    return v_boolean_to_numbers(im_binary)


def cut(im, corners_xy):
    x_list, y_list = corners_xy
    x1, x2 = x_list
    y1, y2 = y_list
    if len(im.shape) == 2:
        return im[int(y1):int(y2), int(x1):int(x2)]
    elif len(im.shape) == 3:
        return im[int(y1):int(y2), int(x1):int(x2), :]


def center_of_mass(im_binary, x_offset=0, y_offset=0):
    """
    Calculates the center of mass of a boolean array.
    True entries are counted with weight 1,
    False entries are counted with weight 0.
    (The optional offsets allows shifting the coordinates.)
    """
    n = np.sum(im_binary)

    x = np.arange(im_binary.shape[1]) + x_offset
    y = np.arange(im_binary.shape[0]) + y_offset
    xv, yv = np.meshgrid(x, y)
    cx = np.sum(xv[im_binary]) / n
    cy = np.sum(yv[im_binary]) / n

    return cx, cy


def recognize_color(color, palette):
    """
    Finds which of the RBG values in palette dict is color
    the closest, and out returns its key.
    """
    min_distance = np.inf
    most_similar_color = None
    for cname, cvalue in palette.items():
        distance = np.sum(np.abs(np.array(color) - np.array(cvalue)))
        if distance < min_distance:
            min_distance = distance
            most_similar_color = cname
    return most_similar_color


def avg_color(im, mask=None):
    if mask is not None:
        return [np.mean(im[:, :, rgb_idx][mask]) for rgb_idx in [0, 1, 2]]
    else:
        [np.mean(im[:, :, rgb_idx]) for rgb_idx in [0, 1, 2]]


def find_shapes(im_bw, kernel, th):
    """
    Finds the centers of mass of the shapes identified to be
    matching with the kernel using convolution and thresholding.
    """
    im_convolved = scipy.ndimage.filters.convolve(im_bw, kernel[::-1, ::-1])
    im_hotspots = im_convolved >= th
    bin_dy, bin_dx = kernel.shape
    radius_x = bin_dx // 2
    radius_y = bin_dy // 2
    positions = []
    open_for_interpretation = np.ones_like(im_convolved, dtype=bool)
    for x in range(im_hotspots.shape[1]):
        for y in range(im_hotspots.shape[0]):
            if open_for_interpretation[y, x] and im_hotspots[y, x]:
                window = im_hotspots[y - radius_y: y + radius_y, x - radius_x: x + radius_x]
                cx, cy = center_of_mass(window, x_offset=x - radius_x, y_offset=y - radius_y)
                positions.append([cx, cy])
                open_for_interpretation[y - radius_y: y + radius_y, x - radius_x: x + radius_x] = False
    return positions
