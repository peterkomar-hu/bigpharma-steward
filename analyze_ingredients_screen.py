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
        self.ingredient_palette = {
            'green': [15, 51, 24],
            'green_max': [83, 86, 28],

            'gray': [73, 86, 90],
            'ligth_gray': [129, 139, 142],
            'gray_conc': [170, 176, 178],

            'crimson': [51, 15, 15],
            'crimson_max': [91, 27, 91],

            'red': [232, 88, 88],
            'red_conc': [244, 177, 177],
            'red_max': [232, 88, 228],
            'red_conc_and_max': [244, 177, 242]
        }
        self.catalyst_palette = {
            'green': [144, 231, 150],
            'blue': [64, 230, 227],
            'purple': [160, 150, 252],
            'orange': [255, 199, 111],
            'pink': [255, 107, 236]}
        with open('./kernels/list-of-basic-cures.txt', 'r') as f_cures:
            known_cures_names = f_cures.read().split('\n')[:-1]

        self.known_cure_kernels = {}
        for cure_name in known_cures_names:
            cure_name_w_hyphens = '-'.join(cure_name.split(' '))
            kernel_file = './kernels/Cure-text-' + cure_name_w_hyphens + '.npy'
            kernel = np.load(kernel_file)
            self.known_cure_kernels[cure_name] = kernel

    def _is_empty(self, effect_box):
        bw = bio.make_bw(effect_box)
        return np.sum(bw > 0) < 50

    def _cut_catalyst(self, effect_box):
        return bio.cut(effect_box, ((240, 290), (0, 65)))

    def _read_catalyst(self, catalyst_box):
        bw = bio.make_bw(catalyst_box)
        mask = bw > 0
        if np.sum(mask) < 50:
            return 'no catalyst'
        else:
            avg_color = bio.avg_color(catalyst_box, mask)
            return bio.recognize_color(avg_color, self. catalyst_palette)

    def _cut_slider(self, effect_box):
        return bio.cut(effect_box, ((0, 240), (65-18, 65)))

    def _read_slider(self, slider):
        """
        Determines min, max and optimal effective concentration,
        using the colors of the teeth of the slider.
        """
        palette = self.ingredient_palette
        colors = []
        for idx in range(0, 20):
            x = idx * 12
            teeth = slider[2:14, x+2:x+9]
            color = np.mean(np.mean(teeth, axis=0), axis=0)
            color_name = bio.recognize_color(color, palette)
            colors.append(color_name)

        if 'green' in colors:
            effect_colors = {'green', 'green_max'}
            max_colors = {'green_max'}
            conc_colors = {'gray_conc'}
            effect_type = 'cure'
        elif 'crimson' in colors:
            effect_colors = {'crimson', 'crimson_max'}
            max_colors = {'crimson_max'}
            conc_colors = {'gray_conc'}
            effect_type = 'side-effect'
        elif 'red' in colors:
            effect_colors = {'red', 'red_max', 'red_conc', 'red_conc_and_max'}
            max_colors = {'red_max', 'red_conc_and_max'}
            conc_colors = {'red_conc', 'red_conc_and_max'}
            effect_type = 'side-effect'
        else:
            return {}

        conc_low = None
        conc_high = None
        conc_optimal = None
        conc_current = None
        for idx, cname in enumerate(colors):
            conc = idx + 1
            if cname in effect_colors:
                conc_high = conc
                if conc_low is None:
                    conc_low = conc
                if cname in max_colors:
                    conc_optimal = conc
            if cname in conc_colors:
                conc_current = conc
        info = {
            'effect_type': effect_type,
            'conc_range': [conc_low, conc_high],
            'conc_optimal': conc_optimal,
            'conc_current': conc_current,
        }
        return info

    def _read_cure_name(self, effect_box):
        """
        Recognize the name of the cure, with the aid of a
        list of known cure names
        """
        cure_name_bw = bio.make_bw(effect_box[15:35, 5:200, :], th=132)
        best_match = None
        min_distance = np.inf
        for cure_name, kernel in self.known_cure_kernels.items():
            distance = np.sum(np.abs(cure_name_bw - kernel))
            if distance < min_distance:
                min_distance = distance
                best_match = cure_name
        return best_match

    def read(self, effect_box):
        if self._is_empty(effect_box):
            info = {'effect_type': 'empty'}
            return info

        slider_box = self._cut_slider(effect_box)
        info = self._read_slider(slider_box)

        catalyst_box = self._cut_catalyst(effect_box)
        catalyst = self._read_catalyst(catalyst_box)
        info['catalyst'] = catalyst

        if info['effect_type'] == 'cure':
            cure_name = self._read_cure_name(effect_box)
            info['cure_name'] = cure_name

        return info


class IngredientBoxReader:
    def __init__(self):
        self.remove_box_reader = RemoveBoxReader()
        self.effect_box_reader = EffectBoxReader()
        self.remove_text_kernel = np.load('./kernels/Remove-text.npy')

    def _find_remove_box(self, ingredient_box):
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

    def _find_unobstructed_effect_boxes(self, ingredient_box, find_remove_box_result):
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

    def _get_slider(self, ingredient_box, effect_idx):
        effect_box_height = 65
        slider_height = 18
        slider_width = 240
        x0 = 280
        x1 = x0 + slider_width
        i = effect_idx
        y0 = 161 + (i + 1) * effect_box_height - slider_height
        y1 = y0 + slider_height
        slider = ingredient_box[y0:y1, x0:x1, :]
        if np.sum(bio.make_bw(slider) > 0) > 50:
            return slider
        else:
            return None

    def read(self, ingredient_box):
        find_remove_box_result = self._find_remove_box(ingredient_box)
        effect_boxes = self._find_unobstructed_effect_boxes(ingredient_box, find_remove_box_result)
        effect_infos = []
        conc_current = None
        for effect_box in effect_boxes:
            if effect_box is None:
                effect_infos.append({})
            else:
                effect_info = self.effect_box_reader.read(effect_box)
                if conc_current is None:
                    if 'conc_current' in effect_info:
                        conc_current = effect_info['conc_current']
                else:
                    if 'conc_current' in effect_info:
                        assert conc_current == effect_info['conc_current']
                effect_infos.append(effect_info)

        if find_remove_box_result is not None:
            remove_box, rmbox_y0, rmbox_y1 = find_remove_box_result

            effect_idx_of_remove = 3
            for i in range(3,0,-1):
                if effect_boxes[i] is None:
                    break
                else:
                    effect_idx_of_remove -= 1

            slider = self._get_slider(ingredient_box, effect_idx_of_remove)
            effect_info_w_remove = self.effect_box_reader._read_slider(slider)

            if remove_box is None:
                effect_info_w_remove['removable'] = False
            else:
                effect_info_w_remove['removable'] = True
                remove_info = self.remove_box_reader.read(remove_box)
                effect_info_w_remove['remove_machine'] = remove_info['machine']
                effect_info_w_remove['remove_conc_range'] = remove_info['conc_range']
            effect_info_w_remove['effect_type'] = 'side-effect'

            effect_infos[effect_idx_of_remove] = effect_info_w_remove
        return effect_infos


def read_ingredients_screen(im):
    """
    Reads all legible information from a screenshot of the Ingredients screen.
     - segments the screen into boxes (ingredients)
     - reads all available information about each box
    Returns a list of lists dicts, where each dict
    represents the information about an effect of the ingredient.
    """
    boxes = cut_boxes(im)

    ingredient_box_reader = IngredientBoxReader()
    ingredients = []
    for box in boxes:
        effect_infos = ingredient_box_reader.read(box)
        ingredients.append(effect_infos)

    return ingredients
