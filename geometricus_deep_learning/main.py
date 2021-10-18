from geometricus_deep_learning.gin import Encoder
from geometricus_deep_learning.losses import local_global_loss_
from geometricus_deep_learning.model import FF, PriorDiscriminator
import torch
import torch.nn as nn


class InfoGraph(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, dataset_num_features, alpha=0.5, beta=1., gamma=.1,
                 prior=False):  # TODO: fix prior
        super(InfoGraph, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prior = prior

        self.embedding_dim = mi_units = hidden_dim * num_gc_layers
        self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)

        self.local_d = FF(self.embedding_dim)
        self.global_d = FF(self.embedding_dim)
        # self.local_d = MI1x1ConvNet(self.embedding_dim, mi_units)
        # self.global_d = MIFCNet(self.embedding_dim, mi_units)

        if self.prior:
            self.prior_d = PriorDiscriminator(self.embedding_dim)

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch, num_graphs):
        # batch_size = data.num_graphs
        if x is None:
            x = torch.ones(batch.shape[0]).to("cuda")

        y, M = self.encoder(x, edge_index, batch)

        g_enc = self.global_d(y)
        l_enc = self.local_d(M)

        mode = 'fd'
        measure = 'JSD'
        local_global_loss = local_global_loss_(l_enc, g_enc, edge_index, batch, measure)

        if self.prior:
            prior = torch.rand_like(y)
            term_a = torch.log(self.prior_d(prior)).mean()
            term_b = torch.log(1.0 - self.prior_d(y)).mean()
            PRIOR = - (term_a + term_b) * self.gamma
        else:
            PRIOR = 0

        return local_global_loss + PRIOR


def train(dataloader, dataset_num_features, hidden_dim, num_gc_layers=3,
          epochs=20, log_interval=1, lr=0.001):
    print('================')
    print('lr: {}'.format(lr))
    print('num_features: {}'.format(dataset_num_features))
    print('hidden_dim: {}'.format(hidden_dim))
    print('num_gc_layers: {}'.format(num_gc_layers))
    print('================')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = InfoGraph(hidden_dim,
                      num_gc_layers,
                      dataset_num_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.eval()
    emb, y, ids = model.encoder.get_embeddings(dataloader)
    print('===== Before training =====')

    for epoch in range(1, epochs + 1):
        loss_all = 0
        model.train()
        for data in dataloader:
            data = data.to(device)
            optimizer.zero_grad()
            loss = model(data.x, data.edge_index, data.batch, data.num_graphs)
            loss_all += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()
        print('===== Epoch {}, Loss {} ====='.format(epoch, loss_all / len(dataloader)))

        if epoch % log_interval == 0:
            model.eval()
            emb, y, ids = model.encoder.get_embeddings(dataloader)

    model.eval()
    emb, y, ids = model.encoder.get_embeddings(dataloader)
    return model, emb, y, ids


if __name__ == '__main__':
    pass
