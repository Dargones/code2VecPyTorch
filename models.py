from modules import *


class Code2VecEncoder(nn.Module):

    def __init__(self, dim, token_voc_size, path_voc_size, properties=0):
        super(Code2VecEncoder, self).__init__()
        self.embed_path = PathEmbeddingModule(dim, token_voc_size, path_voc_size)
        self.attention = AttentionModule(dim)
        self.W = nn.Linear(3 * dim + properties, dim)
        self.properties = properties

    def forward(self, x):
        start, path, end, mask = x[:, 0, :].long(), x[:, 1, :].long(), x[:, 2, :].long(), x[:, 3, :].long()
        embed = self.embed_path(start, path, end)
        if self.properties != 0:
            properties = x[:, 4:, :].permute(0, 2, 1)
            embed = tt.cat((embed, properties), dim=2)
        x = self.W(embed)
        x = tt.tanh(x)
        return self.attention(x, mask)


class BaseCode2Vec(nn.Module):
    """The code2vec model as described in the original paper"""

    def __init__(self, dim, token_voc_size, path_voc_size, target_voc_size, properties=0):
        super(BaseCode2Vec, self).__init__()
        self.encoder = Code2VecEncoder(dim, token_voc_size, path_voc_size, properties)
        self.prediction = NamePredictionModule(dim, target_voc_size)

    def forward(self, x):
        x = self.encoder(x)
        return self.prediction(x)


class ContrastiveCode2Vec(nn.Module):
    """A contrastive version of the code2vec model"""

    # TODO
