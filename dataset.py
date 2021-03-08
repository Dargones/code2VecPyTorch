from torch.utils.data import IterableDataset, Dataset
import torch as tt
import random

from vocabularies import Vocab
from config import Config
from log_tools import get_logger

logger = get_logger(__name__)


class AbstractC2VDataset(Dataset):
    """ An abstract C2V Dataset classes with basic preprocessing utilities included """

    NUM_EXAMPLES_FILE = ".num_examples"

    def __init__(self, filename, token_vocab, path_vocab, target_vocab, properties=0):
        self.filename = filename
        self.token_vocab = token_vocab
        self.path_vocab = path_vocab
        self.target_vocab = target_vocab
        self.properties = properties
        with open(filename + AbstractC2VDataset.NUM_EXAMPLES_FILE) as file:
            lines = file.readlines()
        self.length = int(lines[0].strip("\n"))

    def __len__(self):
        return self.length

    def vectorize(self, line):
        contexts, starts, paths, ends, properties, mask, target = self.init_context(line)
        for i, context in enumerate(contexts[1:]):
            data = context.split(",")
            starts[i] = self.token_vocab.get_ind(data[0])
            paths[i] = self.path_vocab.get_ind(data[1])
            ends[i] = self.token_vocab.get_ind(data[2])
            properties[i] = tt.tensor([float(x) for x in data[3:3 + self.properties]])
        properties = properties.permute((1, 0))
        if self.properties == 0:
            return tt.stack((starts, paths, ends, mask)), target
        return tt.cat((starts.unsqueeze(0), paths.unsqueeze(0), ends.unsqueeze(0), mask.unsqueeze(0), properties)), \
               target

    def init_context(self, line):
        contexts = [c for c in line.strip("\n").split(" ") if c != ""]
        starts = tt.full((Config.MAX_CONTEXTS,), self.token_vocab.get_ind(Vocab.OOV), dtype=tt.long)
        paths = tt.full((Config.MAX_CONTEXTS,), self.path_vocab.get_ind(Vocab.OOV), dtype=tt.long)
        ends = tt.full((Config.MAX_CONTEXTS,), self.token_vocab.get_ind(Vocab.OOV), dtype=tt.long)
        properties = tt.zeros((Config.MAX_CONTEXTS, self.properties), dtype=tt.float)
        mask = tt.zeros(Config.MAX_CONTEXTS, dtype=tt.long)
        mask[:len(contexts) - 1] = 1
        target = tt.tensor(self.target_vocab.get_ind(contexts[0]), dtype=tt.long)
        return contexts, starts, paths, ends, properties, mask, target

    def __iter__(self):
        pass

    def __getitem__(self, index):
        pass


class BaseC2VDataset(AbstractC2VDataset):

    def __init__(self, filename, token_vocab, path_vocab, target_vocab):
        super().__init__(filename, token_vocab, path_vocab, target_vocab)
        # TODO

    def __getitem__(self, index):
        pass
        # TODO


class IterableBaseC2VDataset(AbstractC2VDataset, IterableDataset):

    def __iter__(self):
        file_iterator = open(self.filename)
        mapped_iterator = map(self.vectorize, file_iterator)
        return mapped_iterator


class ShuffleDataset(IterableDataset):
    """
    A shuffling buffer implementation for iterable datasets copied from:
    https://discuss.pytorch.org/t/how-to-shuffle-an-iterable-dataset/64130/5
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