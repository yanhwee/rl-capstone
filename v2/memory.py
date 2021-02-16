class Memory:
    def __init__(self, maxlen):
        self.values = [None] * maxlen
    def __getitem__(self, key):
        assert(self.values[key % len(self.values)] is not None)
        return self.values[key % len(self.values)]
    def __setitem__(self, key, value):
        self.values[key % len(self.values)] = value