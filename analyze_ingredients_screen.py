import numpy as np
import basic_image_operations as bio

from skimage import io
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'
import cv2
from difflib import SequenceMatcher


def cut_boxes(im):
    """
    Locates ingredient boxes and process boxes
    """
    main_im = bio.cut_main_screen(im)
    bw_im = bio.make_bw(main_im)

    kernel = np.load('./kernels/Import-text.npy')
    import_positions = bio.find_shapes(bw_im, kernel, th=6000)

    kernel = np.load('./kernels/Upgrade-text.npy')
    upgrade_positions = bio.find_shapes(bw_im, kernel, th=1950)

    x_offsets = {'import': -512,
                 'upgrade': -145}
    y_offsets = {'import': -80,
                 'upgrade': -395}
    if len(import_positions) >= len(upgrade_positions):
        x_offset = x_offsets['import']
        y_offset = y_offsets['import']
        positions = import_positions
    else:
        x_offset = x_offsets['upgrade']
        y_offset = y_offsets['upgrade']
        positions = upgrade_positions

    width = 580
    height = 430
    boxes = []
    for pos in positions:
        x0 = int(pos[0] + x_offset)
        y0 = int(pos[1] + y_offset)
        x1 = x0 + width
        y1 = y0 + height
        if x0 < 0 or y0 < 0 or \
           x1 > main_im.shape[1] or y1 > main_im.shape[0]:
            continue
        box = main_im[y0:y1, x0:x1, :]
        boxes.append(box)

    return boxes
