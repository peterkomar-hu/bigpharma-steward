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

    def update_from_dict(self, d_raw):
        valid_attributes = self.__dict__.keys()
        d_sanitized = {}
        for k in valid_attributes:
            if k in d_raw:
                d_sanitized[k] = d_raw[k]
        self.__dict__.update(d_sanitized)

    def is_consistent_with(self, other):
        if self.type != other.type:
            return False
        d_self = self.__dict__
        d_other = other.__dict__
        for key in d_self:
            if d_self[key] != d_other[key] and \
               d_self is not None and \
               d_other is not None:
                return False
        return True


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

    def update_from_dict_list(self, dict_list):
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

                new_effect.update_from_dict(effect_dict)
                self.effects[idx] = new_effect

                if self.conc is None and 'conc_current' in effect_dict:
                    self.conc = effect_dict['conc_current']

            else:
                self.effects[idx].update_from_dict(effect_dict)

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

    def id_match(self, id_info_dict):
        mandatory_keys = ['conc_current', 'conc_range', 'effect_type', 'effect_idx']
        for key in mandatory_keys:
            assert key in id_info_dict
        if id_info_dict['conc_current'] != self.conc:
            return False
        idx = id_info_dict['effect_idx']
        stored_effect = self.effects[idx]
        if id_info_dict['effect_type'] != stored_effect.type:
            return False
        if id_info_dict['conc_range'] != stored_effect.conc_range:
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
