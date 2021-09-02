import random

import numpy as np
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
from read_dataset import Dataset

dataset = Dataset(name="AIDS", device="cpu", load_from_disk=False)

uu = [x for x in dataset.dataset if x.x.shape[0]>=40]
dd = uu[3]

def random_color():
    color = tuple(x/255.0 for x in (np.random.choice(range(256), size=3)))
    return color

node_attrs = [''.join(map(lambda x: str(int(x)),x.tolist())) for x in dd.x]
edge_attrs = [''.join(map(lambda x: str(int(x)),x.tolist())) for x in dd.edge_attr]

ddn = to_networkx(dd,to_undirected=True)

isgt_dict1 = dict(zip(ddn.nodes, node_attrs))
isgt_dict2 = dict(zip(ddn.edges, edge_attrs))

nx.set_node_attributes(ddn, isgt_dict1, 'label')
nx.set_edge_attributes(ddn, isgt_dict2, 'label')

n_colors_dict = {i:random_color() for i in (set(node_attrs))}
e_colors_dict = {i:random_color() for i in (set(edge_attrs))}
n_colors = [n_colors_dict[ddn.nodes[i]['label']] for i in ddn.nodes()]
e_colors = [e_colors_dict[ddn.edges[i]['label']] for i in ddn.edges()]

pos = nx.kamada_kawai_layout(ddn)
nx.draw(ddn, pos=pos, with_labels=False, node_color=n_colors,edge_color=e_colors, width=5)

plt.show()