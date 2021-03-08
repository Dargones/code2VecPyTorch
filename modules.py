import torch as tt
import torch.nn as nn
import torch.nn.functional as F

# TODO: Code2vec uses "normalized initialization" from Glorot and Bengio (2010). Should\Could this be replicated here?


class AttentionModule(nn.Module):
    """A module for combining path vectors into a code vector with attention"""

    def __init__(self, dim):
        super(AttentionModule, self).__init__()
        self.att = nn.Parameter(tt.randn(1, dim, 1))  # learnable attention

    def forward(self, x, mask):
        """
        :param x:    [BATCH, MAX_PATHS, DIM]
        :param mask: [BATCH, MAX_PATHS]
        :return:     [BATCH, DIM]
        """
        att = self.att.repeat(x.shape[0], 1, 1)  # [BATCH, DIM, 1]
        att_weights = tt.bmm(x, att).squeeze(2)  # [BATCH, MAX_PATHS]
        att_weights[mask == 0] = float("-inf")   # set "padded" zeros to -inf
        att_weights = F.softmax(att_weights, 1)  # [BATCH, MAX_PATHS]
        att_weights = att_weights.unsqueeze(2)   # [BATCH, MAX_PATHS, 1]
        return tt.sum((x * att_weights), dim=1)   # [BATCH, MAX_PATHS]


class NamePredictionModule(nn.Module):
    """
    A module for predicting labels from the code vector.
    Label embeddings are learned in the process.
    """

    def __init__(self, dim, label_voc_size):
        super(NamePredictionModule, self).__init__()
        self.le = nn.Parameter(tt.randn(dim, label_voc_size))  # embeddings

    def forward(self, x):
        """
        :param x:    [BATCH, DIM]
        :return:     [BATCH, N_LABELS]
        """
        x = tt.mm(x, self.le)  # [BATCH, N_LABELS]
        return F.softmax(x, dim=1)


class PathEmbeddingModule(nn.Module):
    """A module for embedding an AST path"""

    def __init__(self, dim, token_voc_size, path_voc_size):
        super(PathEmbeddingModule, self).__init__()
        self.pe = nn.Embedding(path_voc_size, dim)
        self.te = nn.Embedding(token_voc_size, dim)

    def forward(self, start, path, end):
        """
        :param start: [BATCH, MAX_PATHS, 1]
        :param end:   [BATCH, MAX_PATHS, 1]
        :param path:  [BATCH, MAX_PATHS, 1]
        :return:      [BATCH, MAX_PATHS, DIM]
        """
        start_e = self.te(start)
        path_e = self.pe(path)
        end_e = self.te(end)
        return tt.cat((start_e, path_e, end_e), dim=2)
