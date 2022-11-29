"""Model zoo with all flavors of graph neural network layers."""
from functools import partial
import dgl

__ALL__ = ["GCN", "GraphSAGE", "GAT", "GATDot", "PNA"]

GCN = partial(dgl.nn.GraphConv, allow_zero_in_degree=True)
GCN.__doc__ = dgl.nn.GraphConv.__doc__

GraphSAGE = partial(dgl.nn.SAGEConv, aggregator_type="mean")
GraphSAGE.__doc__ = dgl.nn.SAGEConv.__doc__

PNA = partial(
    dgl.nn.PNAConv,
    aggregators=["mean", "max", "min", "std", "var", "sum"],
    scalers=["identity", "amplification"],
    delta=3.0,
)

from .gat import GAT, GATDot
