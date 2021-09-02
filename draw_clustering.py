import random

import numpy as np
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
from read_dataset import Dataset

dataset = Dataset(name="AIDS", device="cpu", load_from_disk=False)

# bbb = [x for x in fin[:,0]]
# ccc = [x for x in fin[:,1]]

yy = np.array(dataset.y)

c1_axis = fin[(kmeans == 0).nonzero()]
c2_axis = fin[(kmeans == 1).nonzero()]

plt.plot([x[0] for x in c1_axis],[x[1] for x in c1_axis],'o',color='b')
plt.plot([x[0] for x in c2_axis],[x[1] for x in c2_axis],'o',color='r')

plt.show()