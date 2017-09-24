import analyze_cures_screen as cures


class ReactionPipeline:
    def __init__(self):
        self.cures = []
        self.processes = []


class Cure:
    def __init__(self):
        self.name = None
        self.price = None
        self.active_conc_range = [None, None]
        self.optimal_conc = None
        self.prev_cure = None
        self.next_cure = None
        self.prev_process = None
        self.next_process = None

    def update(self, info):
        if 'name' in info:
            if info['name']: self.name = info['name']
        if 'price' in info:
            if info['price']: self.price = info['price']
        if 'conc_range' in info:
            new_range = info['conc_range']
            for idx, limit in enumerate(new_range):
                if limit: self.active_conc_range[idx] = new_range
        if 'conc_optimal' in info:
            if info['conc_optimal']: self.optimal_conc = info['conc_optimal']


class Process:
    def __init__(self):
        self.machine = None
        self.conc_range = [None, None]
        self.catalyst = None

    def update(self, info):
        """
        Updates the attributes based on the non-None values of info dict
        """
        if 'machine' in info:
            new_machine = info['machine']
            if new_machine: self.machine = new_machine

        if 'conc_range' in info:
            new_range = info['conc_range']
            for idx, limit in enumerate(new_range):
                if limit: self.conc_range[idx] = new_range

        if 'catalyst' in info:
            new_catalyst = info['catalyst']
            if new_catalyst: self.catalyst = new_catalyst

    def is_overlapping(self, other):
        return self.machine != other.machine and \
               self.conc_range[0] < other.conc_range[1] and \
               self.conc_range[1] < other.conc_range[0]

