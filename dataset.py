from config import *
from torch.utils.data import IterableDataset
import torch as tt
import random

from vocabularies import Vocab


class ShuffleDataset(IterableDataset):
    """
    This allows us to have a shuffling buffer.
    Copied from https://discuss.pytorch.org/t/how-to-shuffle-an-iterable-dataset/64130/5
    """

    def __init__(self, dataset, buffer_size):
        super().__init__()
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        shufbuf = []
        try:
            dataset_iter = iter(self.dataset)
            for i in range(self.buffer_size):
                shufbuf.append(next(dataset_iter))
        except:
            self.buffer_size = len(shufbuf)

        try:
            while True:
                try:
                    item = next(dataset_iter)
                    evict_idx = random.randint(0, self.buffer_size - 1)
                    yield shufbuf[evict_idx]
                    shufbuf[evict_idx] = item
                except StopIteration:
                    break
            while len(shufbuf) > 0:
                yield shufbuf.pop()
        except GeneratorExit:
            pass


class Code2VecDataset(IterableDataset):
    """
    An iterable dataset for code2vec files. Modified version of:
    https://medium.com/swlh/how-to-use-pytorch-dataloaders-to-work-with-enormously-large-text-files-bbd672e955a0
    """

    def __init__(self, filename, token_vocab, path_vocab, target_vocab):
        self.filename = filename
        self.token_vocab = token_vocab
        self.path_vocab = path_vocab
        self.target_vocab = target_vocab
        self.length = None

    def __len__(self):
        if not self.length:
            with open(self.filename + ".num_examples") as file:
                lines = file.readlines()
            self.length = int(lines[0].strip("\n"))
        return self.length

    def line_mapper(self, line):
        contexts, starts, paths, ends, mask, target = self.init_context(line)
        for i, context in enumerate(contexts[1:]):
            data = context.split(",")
            starts[i] = self.token_vocab.get_ind(data[0])
            paths[i] = self.path_vocab.get_ind(data[1])
            ends[i] = self.token_vocab.get_ind(data[2])
        return tt.stack((starts, paths, ends, mask)), target

    def init_context(self, line):
        contexts = [c for c in line.strip("\n").split(" ") if c != ""]
        target = tt.tensor(self.target_vocab.get_ind(contexts[0]), dtype=tt.long)
        starts = tt.full((MAX_CONTEXTS,), self.token_vocab.get_ind(Vocab.PAD), dtype=tt.long)
        paths = tt.full((MAX_CONTEXTS,), self.path_vocab.get_ind(Vocab.PAD), dtype=tt.long)
        ends = tt.full((MAX_CONTEXTS,), self.token_vocab.get_ind(Vocab.PAD), dtype=tt.long)
        mask = tt.zeros(MAX_CONTEXTS, dtype=tt.long)
        mask[:len(contexts) - 1] = 1
        return contexts, starts, paths, ends, mask, target

    def __iter__(self):
        file_itr = open(self.filename)
        mapped_itr = map(self.line_mapper, file_itr)
        return mapped_itr


class SmartCode2VecDataset(Code2VecDataset):

    def __init__(self, filename, token_vocab, path_vocab, target_vocab,
                 properties_dim):
        super().__init__(filename, token_vocab, path_vocab, target_vocab)
        self.properties_dim = properties_dim

    def line_mapper(self, line):
        contexts, starts, paths, ends, mask, target = self.init_context(line)
        properties = tt.zeros((MAX_CONTEXTS, self.properties_dim), dtype=tt.double)

        for i, context in enumerate(contexts[1:]):
            data = context.split(",")
            starts[i] = self.token_vocab.get_ind(data[0])
            paths[i] = self.path_vocab.get_ind(data[1])
            ends[i] = self.token_vocab.get_ind(data[2])
            properties[i] = [float(x) for x in data[3:]]

        return tt.cat((starts.unsqueeze(0), paths.unsqueeze(0),
                       ends.unsqueeze(0), mask.unsqueeze(0), properties)), \
               target