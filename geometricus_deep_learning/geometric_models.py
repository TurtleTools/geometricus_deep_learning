from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
from torch_geometric.nn import global_mean_pool
import torch


class GCN(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, aggregator="mean"):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(input_channels, hidden_channels, aggr=aggregator)
        self.conv2 = GraphConv(hidden_channels, hidden_channels, aggr=aggregator)
        self.conv3 = GraphConv(hidden_channels, hidden_channels, aggr=aggregator)
        self.lin = Linear(hidden_channels, output_channels)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


class Simem(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, aggregator="mean"):
        super(Simem, self).__init__()
        self.conv1 = GraphConv(input_channels, hidden_channels, aggr=aggregator)
        self.conv2 = GraphConv(hidden_channels, hidden_channels, aggr=aggregator)
        self.conv3 = GraphConv(hidden_channels, hidden_channels, aggr=aggregator)
        self.lin = Linear(hidden_channels, output_channels)

    def forward_single(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x

    def forward(self, x1, x2, edge_index1, edge_index2, batch):
        x1 = self.forward_single(x1, edge_index1, batch)
        x2 = self.forward_single(x2, edge_index2, batch)
        return x1, x2
