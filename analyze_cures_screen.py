import numpy as np
import basic_image_operations as bio

from skimage import io
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'
import cv2
from difflib import SequenceMatcher


def identify_gap(gap, cure_box_gap, process_box_gap):
    if abs(gap - cure_box_gap) < 10:
        return 'cure'
    if abs(gap - process_box_gap) < 10:
        return 'process'
    else:
        return 'gap'


def cut_boxes(im):
    """
    Locates effect boxes and process boxes on the b&w
    screen shot of the Cures screen.
    """
    im_main_screen = bio.cut_main_screen(im)
    im_bw = bio.make_bw(im_main_screen)

    # find triangles
    kernel = np.load('./kernels/triangle.npy')
    triangle_centers = bio.find_shapes(im_bw, kernel, 950)
    triangle_centers.sort()

    # group triangles into columns
    columns_of_triangles = []
    columns_of_triangles.append([triangle_centers[0]])
    for triangle in triangle_centers[1:]:
        if np.abs(columns_of_triangles[-1][0][0] - triangle[0]) < 50:
            columns_of_triangles[-1].append(triangle)
        else:
            columns_of_triangles.append([triangle])

    cure_box_height = 63
    process_box_height = 125
    box_width = 290
    padding = 15
    last_padding = 22

    cure_box_gap = cure_box_height + 2 * padding
    process_box_gap = process_box_height + 2 * padding

    columns_of_box_identities = []  # 'drug' or 'process'
    columns_of_box_coordinates = []  # (x1, x2, y1, y2)
    for idx, col in enumerate(columns_of_triangles):
        if len(col) < 2:
            print('Warning: missing triangles in column ' + str(idx))
            break

        box_identities = []
        box_coordinates = []
        columns_of_box_identities.append(box_identities)
        columns_of_box_coordinates.append(box_coordinates)
        for i in range(len(col) - 1):

            gap = col[i + 1][1] - col[i][1]
            gap_identity = identify_gap(gap, cure_box_gap, process_box_gap)
            if gap_identity in ['cure', 'process']:
                box_identities.append(gap_identity)
                y1 = col[i][1] + padding
                y2 = col[i + 1][1] - padding
                x1 = col[i][0] - box_width // 2
                x2 = col[i][0] + box_width // 2
                box_coordinates.append((x1, x2, y1, y2))

            else:
                box_identities.append('cure')
                x, y = col[i]
                y1 = y + last_padding
                y2 = y + last_padding + cure_box_height
                x1 = x - box_width // 2
                x2 = x + box_width // 2
                box_coordinates.append((x1, x2, y1, y2))

                box_identities.append('cure')
                x, y = col[i + 1]
                y1 = y - padding - cure_box_height
                y2 = y - padding
                x1 = x - box_width // 2
                x2 = x + box_width // 2
                box_coordinates.append((x1, x2, y1, y2))

        # revise first box revision
        if box_identities[0] == 'cure':
            box_identity = 'process'
            box_height = process_box_height
        elif box_identities[0] == 'process':
            box_identity = 'cure'
            box_height = cure_box_height
        x, y = col[0]
        y1 = y - padding - box_height
        y2 = y - padding
        x1 = x - box_width // 2
        x2 = x + box_width // 2
        if y1 > 0 and y2 < im_main_screen.shape[0]:
            box_identities.insert(0, box_identity)
            box_coordinates.insert(0, (x1, x2, y1, y2))

        # revise last box
        if box_identities[-1] == 'cure':
            box_identity = 'process'
            box_height = process_box_height
        elif box_identities[-1] == 'process':
            box_identity = 'cure'
            box_height = cure_box_height
        x, y = col[-1]
        y1 = y + last_padding
        y2 = y + last_padding + box_height
        x1 = x - box_width // 2
        x2 = x + box_width // 2
        if y1 > 0 and y2 < im_main_screen.shape[0]:
            box_identities.append(box_identity)
            box_coordinates.append((x1, x2, y1, y2))

    # cut from image
    columns_of_boxes = []
    for box_coordinates in columns_of_box_coordinates:
        boxes = []
        columns_of_boxes.append(boxes)
        for box_coord in box_coordinates:
            x1, x2, y1, y2 = box_coord
            boxes.append(im_main_screen[int(y1):int(y2), int(x1):int(x2), :])

    return columns_of_boxes, columns_of_box_identities


def read_cure_name(cure_box, known_cures, tmp_dir='./tmp/', th=167):
    """
    Recognize the name of the cure, with the aid of a
    list of known cure names
    """
    tmp_path = tmp_dir + 'tmp.png'
    tmp_gray_path = tmp_dir + 'tmp-gray.png'
    im_segment = cure_box[:, :200, :]
    io.imsave(tmp_path, im_segment)
    colored = cv2.imread(tmp_path)
    gray = cv2.cvtColor(colored, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite(tmp_gray_path, gray)

    raw_text = pytesseract.image_to_string(Image.open(tmp_gray_path))

    matches = []
    for cure in known_cures:
        match_score = SequenceMatcher(None, raw_text, cure).ratio()
        matches.append((match_score, cure))
    best_score, best_text = max(matches, key=lambda t: t[0])
    return raw_text, best_text, best_score


def cut_price(box):
    """
    Selects the part of the image that contains
    the price of the cure after the dollar sign
    """
    kernel = np.load('./kernels/dollar.npy')
    th = 650
    bw = bio.make_bw(box)

    dollar_centers = bio.find_shapes(bw, kernel, th)
    if len(dollar_centers) != 1:
        return None
    x, y = dollar_centers[0]
    height = 20
    padding = 10
    return box[int(y) - height // 2:int(y) + height // 2, int(x) + padding:]


def cut_concentration(process_box):
    """
    Selects the part of the image that contain concentration range,
    using the position of the flask icon in a process box,
    from the Cures screen.
    """
    bw = bio.make_bw(process_box)
    kernel = np.load('./kernels/flask.npy')
    flask_center = bio.find_shapes(bw, kernel, 500)
    if len(flask_center) != 1:
        return None
    x, y = flask_center[0]
    height = 20
    padding = 5
    length = 60
    return process_box[
            int(y) - height // 2 : int(y) + height // 2 + 2,
            int(x) - length - padding : int(x) - padding]


def read_integer(im_segment, tmp_dir='./tmp/', th=200):
    """
    Recognize a single integer in a segmented image.
    """
    tmp_path = tmp_dir + 'tmp_number.png'
    tmp_gray_path = tmp_dir + 'tmp_number-gray.png'
    io.imsave(tmp_path, im_segment)
    colored = cv2.imread(tmp_path)
    gray = cv2.cvtColor(colored, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)[1]

    cv2.imwrite(tmp_gray_path, gray)

    text = pytesseract.image_to_string(Image.open(tmp_gray_path),
                                       config='-psm 6 digits')
    list_of_digits = []
    for c in text:
        if c.isdigit():
            list_of_digits.append(c)

    try:
        number = int(''.join(list_of_digits))
    except ValueError:
        number = None
    return number


def read_integer_range(im_segment, tmp_dir='./tmp/', th=150):
    """
    Recognizes an integer range, such as "13-17"
    in a segmented image.
    """
    tmp_path = tmp_dir + 'tmp_number.png'
    tmp_gray_path = tmp_dir + 'tmp_number-gray.png'
    io.imsave(tmp_path, im_segment)
    colored = cv2.imread(tmp_path)
    gray = cv2.cvtColor(colored, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)[1]

    cv2.imwrite(tmp_gray_path, gray)

    text = pytesseract.image_to_string(Image.open(tmp_gray_path),
                                       config='-psm 6 digits')
    lists_of_digits = text.split('-')
    if len(lists_of_digits) != 2:
        return [None, None]
    numbers = []
    for digits in lists_of_digits:
        try:
            number = int(''.join(digits))
        except ValueError:
            number = None
        numbers.append(number)
    return numbers


def read_cure_price(cure_box, tmp_dir='./tmp/', th=200):
    price_box = cut_price(cure_box)
    return read_integer(price_box, tmp_dir=tmp_dir, th=th)


def read_concentration(process_box):
    conc_box = cut_concentration(process_box)
    return read_integer_range(conc_box)


def read_cure_slider(slider):
    """
    Determines min, max and optimal effective concentration,
    using the colors of the teeth of the slider.
    """
    palette = {
        'gray': [139, 148, 151],
        'yellow': [250, 255, 156],
        'lighter_blue': [99, 255, 252],
        'green': [101, 234, 108],
        'blue': [102, 236, 231],
        'purple': [214, 166, 255],
        'orange': [254, 167, 89]
    }
    colors = []
    for idx in range(0, 20):
        x = idx * 12
        teeth = slider[2:14, x + 2:x + 9]
        color = np.mean(np.mean(teeth, axis=0), axis=0)
        color_name = bio.recognize_color(color, palette)
        colors.append(color_name)
    if 'green' in colors:
        del (palette['blue'])
    else:
        del (palette['lighter_blue'])
    colors = []
    for idx in range(0, 20, 1):
        x = idx * 12
        teeth = slider[2:14, x + 2:x + 9]
        color = np.mean(np.mean(teeth, axis=0), axis=0)
        color_name = bio.recognize_color(color, palette)
        colors.append(color_name)

    effect_colors = ['green', 'blue', 'purple', 'orange', 'yellow']
    conc_low = None
    conc_optimal = None
    conc_high = None
    for idx, cname in enumerate(colors):
        conc = idx + 1
        if cname in effect_colors:
            conc_high = conc
            if conc_low is None:
                conc_low = conc
        if cname == 'yellow':
            conc_optimal = conc

    return [conc_low, conc_high], conc_optimal


def read_hoover_cure_concentrations(cure_box):
    box_bw = bio.make_bw(cure_box)

    active_range_text_kernel = np.load('./kernels/Active-Range-text.npy')
    active_range_text_position = bio.find_shapes(box_bw, active_range_text_kernel, 1850)
    assert len(active_range_text_position) == 1

    max_text_kernel = np.load('./kernels/Max-text.npy')
    max_text_position = bio.find_shapes(box_bw, max_text_kernel, 700)
    assert len(max_text_position) == 1

    flask_kernel = np.load('./kernels/flask.npy')
    flask_positions = bio.find_shapes(box_bw, flask_kernel, 500)
    assert len(flask_positions) == 2

    arx, ary = active_range_text_position[0]
    mx, my = max_text_position[0]
    f1x, f1y = flask_positions[0]
    f2x, f2y = flask_positions[1]

    range_coordinates = [[arx + 50, f1x - 12], [ary - 10, ary + 10]]
    max_coordinates = [[mx + 20, f2x - 12], [my - 10, my + 10]]
    slider_coordinates = [[arx - 51, arx + 189], [ary + 25, ary + 42]]

    conc_range = read_integer_range(bio.cut(cure_box, range_coordinates))
    conc_optimal = read_integer(bio.cut(cure_box, max_coordinates))

    conc_range_slider, conc_optimal_slider = read_cure_slider(bio.cut(cure_box, slider_coordinates))

    if conc_range[0] != conc_range_slider[0]:
        conc_range[0] = None
    if conc_range[1] != conc_range_slider[1]:
        conc_range[1] = None
    if conc_optimal != conc_optimal_slider:
        conc_optimal = None

    return conc_range, conc_optimal


def analyze_cure_box(cure_box, known_cures):
    """
    Determines if the cure box is hoovered over or not,
    and recognizes the information contained within.
    """
    info = {}
    price_box = cut_price(cure_box)
    if price_box is not None:
        info['price'] = read_integer(price_box)
        _, name, _ = read_cure_name(cure_box, known_cures)
        info['name'] = name
    else:
        conc_range, conc_optimal = read_hoover_cure_concentrations(cure_box)
        info['conc_range'] = tuple(conc_range)
        info['conc_optimal'] = conc_optimal

    return info
