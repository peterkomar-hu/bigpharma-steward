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
