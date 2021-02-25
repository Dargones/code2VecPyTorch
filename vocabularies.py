import pickle


class Vocab:

    OOV = "<OOV>"
    PAD = "<PAD>"

    def __init__(self, file, max_tokens):
        freqs = pickle.load(file)
        keys = sorted(freqs.keys(), key=lambda x: freqs[x], reverse=True)
        self.ind_to_key = keys[:max_tokens - 2] + [Vocab.OOV, Vocab.PAD]
        self.key_to_ind = {key: i for i, key in enumerate(self.ind_to_key)}

    def get_ind(self, key):
        if key not in self.key_to_ind:
            return self.key_to_ind[Vocab.OOV]
        return self.key_to_ind[key]

    def get_key(self, ind):
        return self.ind_to_key[ind]