import json
import basic_image_operations as bio
import analyze_ingredients_screen as ais

class Effect:
    def __init__(self, effect_type='empty'):
        self.type = effect_type
        if self.type != 'empty':
            self.conc_range = [None, None]
            self.conc_optimal = None

    def missing_entries(self):
        missing = []
        if self.type != 'empty':
            if self.conc_optimal is None:
                missing.append('conc_optimal')
            for idx in [0, 1]:
                if self.conc_range[idx] is None:
                    missing.append('conc_range[{}]'.format(idx))

        return missing

    def load_from_dict(self, d_raw):
        valid_attributes = self.__dict__.keys()
        d_sanitized = {}
        for k in valid_attributes:
            if k in d_raw:
                d_sanitized[k] = d_raw[k]
        self.__dict__.update(d_sanitized)

    def is_consistent_with(self, other):
        d = self.__dict__
        d_other = other.__dict__
        for k in d:
            try:
                for idx, item in enumerate(d[k]):
                    other_item = d_other[k][idx]
                    if item is not None and other_item is not None:
                        if d[k][idx] != d_other[k][idx]:
                            return False
            except TypeError:
                if d[k] is not None and d_other[k] is not None:
                    if d[k] != d_other[k]:
                        return False
        return True

    def merge(self, other):
        d = self.__dict__
        d_other = other.__dict__
        for k in d:
            try:
                for idx, item in enumerate(d[k]):
                    if item is None:
                        d[k][idx] = d_other[k][idx]
            except TypeError:
                if d[k] is None:
                    d[k] = d_other[k]


class Cure(Effect):
    def __init__(self):
        Effect.__init__(self, effect_type='cure')
        self.cure_name = None

    def missing_entries(self):
        missing = Effect.missing_entries(self)
        if self.cure_name is None:
            missing.append('cure_name')
        return missing


class SideEffect(Effect):
    def __init__(self):
        Effect.__init__(self, effect_type='side-effect')
        self.catalyst = None
        self.removable = None
        self.remove_conc_range = [None, None]
        self.remove_machine = None

    def missing_entries(self):
        missing = Effect.missing_entries(self)
        attributes = self.__dict__
        for attribute_name in ['catalyst', 'removable']:
            value = attributes[attribute_name]
            if value is None:
                missing.append(attribute_name)
        if self.removable == True:
            if self.remove_machine is None:
                missing.append('remove_machine')
            for idx in [0, 1]:
                if self.remove_conc_range[idx] is None:
                    missing.append('remove_conc_range[{}]'.format(idx))
        return missing


class Ingredient:
    def __init__(self):
        self.conc = None
        self.effects = [None, None, None, None]

    def load_from_dict_list(self, dict_list):
        for idx, effect_dict in enumerate(dict_list):
            if 'effect_type' not in effect_dict: continue
            effect_type = effect_dict['effect_type']

            if self.effects[idx] is None:
                if effect_type == 'empty':
                    new_effect = Effect(effect_type='empty')
                elif effect_type == 'cure':
                    new_effect = Cure()
                elif effect_type == 'side-effect':
                    new_effect = SideEffect()

                new_effect.load_from_dict(effect_dict)
                self.effects[idx] = new_effect

                if self.conc is None and 'conc_current' in effect_dict:
                    self.conc = effect_dict['conc_current']

            else:
                self.effects[idx].load_from_dict(effect_dict)

    def missing(self):
        missing_info_list = []
        if self.conc is None:
            missing_info_list.append('conc')
        for idx, effect in enumerate(self.effects):
            if effect is None:
                missing_info_list.append(': '.join(['eff ' + str(idx + 1), 'effect_type']))
                continue
            for missing in effect.missing_entries():
                missing_info_list.append(': '.join(['eff ' + str(idx + 1), missing]))
        return missing_info_list

    def all_effect_types_known(self):
        for effect in self.effects:
            if effect is None:
                return False
            if effect.type is None:
                return False
        return True

    def is_consistent_with(self, other):
        if self.conc is not None and \
                        other.conc is not None and \
                        self.conc != other.conc:
            return False
        for idx in range(4):
            if self.effects[idx] is None:
                continue
            if other.effects[idx] is None:
                continue
            if self.effects[idx].is_consistent_with(other.effects[idx]) == False:
                return False
        return True

    def merge(self, other):
        if self.conc is None:
            self.conc = other.conc
        for idx, _ in enumerate(self.effects):
            if self.effects[idx] is None:
                self.effects[idx] = other.effects[idx]
            elif other.effects[idx] is not None:
                self.effects[idx].merge(other.effects[idx])


class IngredientCollection:
    def __init__(self):
        self.ingredients = []
        self.unmatched_ingredients = []

    def __repr__(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          indent=4, separators=(', ', ': '))

    def load_from_file(self, path):
        pass

    def save_to_file(self, path):
        with open(path, 'w') as fout:
            fout.write(self.__repr__())

    def update_from_screenshot(self, im_path):
        im = bio.load_image(im_path)
        if bio.identify_image(im) != 'ingredients':
            print(im_path + ' is not a screenshot of the Ingredients screen.')
            return None
        for ingredient_data in ais.read_ingredients_screen(im):
            ingr = Ingredient()
            ingr.load_from_dict_list(ingredient_data)

            matching_idx = None
            for idx, stored_ingr in enumerate(self.ingredients):
                if ingr.is_consistent_with(stored_ingr):
                    matching_idx = idx
                    break
            if matching_idx is not None:
                self.ingredients[matching_idx].merge(ingr)
            elif ingr.all_effect_types_known():
                unmatched_remaining = []
                for unmatched_ingr in self.unmatched_ingredients:
                    if ingr.is_consistent_with(unmatched_ingr):
                        ingr.merge(unmatched_ingr)
                    else:
                        unmatched_remaining.append(unmatched_ingr)
                self.unmatched_ingredients = unmatched_remaining
                self.ingredients.append(ingr)

            else:
                self.unmatched_ingredients.append(ingr)
