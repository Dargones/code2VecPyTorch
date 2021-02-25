from modules import *


class Code2Vec(nn.Module):
    """The code2vec model as described in the original paper"""

    def __init__(self, dim, token_voc_size, path_voc_size, label_voc_size):
        super(Code2Vec, self).__init__()
        self.embed_path = PathEmbeddingModule(dim, token_voc_size, path_voc_size)
        self.attention = AttentionModule(dim)
        self.prediction = NamePredictionModule(dim, label_voc_size)
        self.W = nn.Linear(3 * dim, dim)

    def forward(self, X):
        start, path, end, mask = X[:, 0, :], X[:, 1, :], X[:, 2, :], X[:, 3, :]
        x = self.embed_path(start, path, end)
        x = self.W(x)
        x = tt.tanh(x)
        x = self.attention(x, mask)
        return self.prediction(x)


class SmartCode2Vec(Code2Vec):

    def __init__(self, dim, token_voc_size, path_voc_size, label_voc_size, properties_dim):
        super(SmartCode2Vec, self).__init__()
        self.W = nn.Linear(3 * dim + properties_dim, dim)

    def forward(self, X):
        start, path, end, mask = X[:, 0, :], X[:, 1, :], X[:, 2, :], X[:, 3, :]
        properties = X[:, 4:, :].permute(0, 2, 1)
        x = tt.cat((self.embed_path(start, path, end), properties), dim=2)
        x = self.W(x)
        x = tt.tanh(x)
        x = self.attention(x, mask)
        return self.prediction(x)