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
    import_positions = bio.find_shapes(bw_im, kernel, th=5700)

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


def read_digits_by_kernels(segment, list_of_digit_kernels):
    im_bw = bio.make_bw(segment, th=200)
    digits = []
    for digit in range(0, 10, 1):
        kernel = list_of_digit_kernels[digit]
        positions = bio.find_shapes(im_bw, kernel, th=160)
        for pos in positions:
            x = pos[0]
            digits.append((x, digit))
    digits.sort()
    return tuple([d[1] for d in digits])


class RemoveBoxReader:
    def __init__(self):
        self.flask_kernel = np.load('./kernels/flask.npy')
        self.machine_names = [
            'Dissolver',
            'Evaporator',
            'Ioniser',
            'Agglomerator',
            'Autoclave',
            'Cryogenic-Condenser',
            'Sequencer',
            'Chromatograph',
            'Ultraviolet-Curer'
        ]
        self.machine_dict = {}
        for name in self.machine_names:
            kernel = np.load('./kernels/' + name + '-remove.npy')
            self.machine_dict[name] = kernel

        self.conc_digits = []
        for digit in range(0, 10, 1):
            kernel = np.load('./kernels/conc_' + str(digit) + '_remove_kernel.npy')
            self.conc_digits.append(kernel)

    def _cut_machine(self, remove_box):
        return bio.cut(remove_box, ((210, 310), (25, 125)))

    def _recognize_machine(self, remove_box):
        segment = bio.make_bw(self._cut_machine(remove_box))
        min_distance = np.inf
        closest_machine = None
        for name, kernel in self.machine_dict.items():
            distance = np.sum(np.abs(kernel - segment))
            if distance < min_distance:
                min_distance = distance
                closest_machine = name
        return closest_machine

    def _cut_concentration(self, remove_box):
        """
        Selects the part of the image that contain concentration range,
        using the position of the flask icon in a process box,
        from the Cures screen.
        """
        bw = bio.make_bw(remove_box)
        kernel = self.flask_kernel
        flask_center = bio.find_shapes(bw, kernel, 465)
        if len(flask_center) != 1:
            return None
        x, y = flask_center[0]
        height = 20
        padding = 5
        length = 60
        return remove_box[
               int(y) - height // 2: int(y) + height // 2 + 2,
               int(x) - length - padding: int(x) - padding]

    def _read_concentration(self, remove_box):
        conc_box = self._cut_concentration(remove_box)
        digits = read_digits_by_kernels(conc_box, self.conc_digits)
        if len(digits) == 2:
            low = digits[0]
            high = digits[1]
        elif len(digits) == 3:
            low = digits[0]
            high = 10*digits[1] + digits[2]
        elif len(digits) == 4:
            low = 10*digits[0] + digits[1]
            high = 10*digits[2] + digits[3]
        else:
            return None, None
        return low, high

    def read(self, remove_box):
        info = {}
        info['machine'] = self._recognize_machine(remove_box)
        info['conc_range'] = self._read_concentration(remove_box)
        return info


class EffectBoxReader:
    def __init__(self):
        pass


class IngredientBoxReader:
    def __init__(self):
        self.remove_box_reader = RemoveBoxReader()
        self.remove_text_kernel = np.load('./kernels/Remove-text.npy')

    def find_remove_box(self, ingredient_box):
        bw_box = bio.make_bw(ingredient_box)
        positions = bio.find_shapes(bw_box,
                                    self.remove_text_kernel,
                                    th=1100)
        assert len(positions) <= 1, 'At most one "Remove" text is allowed'
        if len(positions) == 0:
            return None
        x = int(positions[0][0])
        y = int(positions[0][1])
        x0 = x - 85
        x1 = x - 30
        y0 = y - 10
        y1 = y + 15
        is_cannot = np.sum(bw_box[y0:y1, x0:x1] > 0) > 100

        if is_cannot:
            upper_edge = y - 24
            lower_edge = y + 24
            remove_box = None
        else:
            x0 = x - 165
            x1 = x + 165
            y0 = y + 15
            y1 = y + 144
            upper_edge = y0 - 31
            lower_edge = y1
            remove_box = bio.cut(ingredient_box, ((x0, x1), (y0, y1)))
        return remove_box, upper_edge, lower_edge

    def find_unobstructed_effect_boxes(self, ingredient_box, find_remove_box_result):
        if find_remove_box_result is not None:
            _, rmbox_y0, rmbox_y1 = find_remove_box_result
        else:
            rmbox_y0 = np.inf
            rmbox_y1 = -np.inf
        effect_box_height = 65
        effect_box_width = 290
        x0 = 280
        x1 = x0 + effect_box_width
        effect_boxes = []
        for i in range(4):
            y0 = 161 + i * effect_box_height
            y1 = y0 + effect_box_height
            if y0 > rmbox_y1 or y1 < rmbox_y0:
                effect_box = ingredient_box[y0:y1, x0:x1, :]
            else:
                effect_box = None
            effect_boxes.append(effect_box)
        return effect_boxes

    def find_unobstructed_slider(self, ingredient_box, rmbox_upper_edge, lower_edge):
        rmbox_y0 = rmbox_upper_edge
        rmbox_y1 = lower_edge
        effect_box_height = 65
        slider_height = 18
        slider_width = 240
        x0 = 280
        x1 = x0 + slider_width
        for i in [0, 3]:
            y0 = 161 + (i+1) * effect_box_height - slider_height
            y1 = y0 + slider_height
            if y0 > rmbox_y1 or y1 < rmbox_y0:
                slider = ingredient_box[y0:y1, x0:x1, :]
                return i, slider
