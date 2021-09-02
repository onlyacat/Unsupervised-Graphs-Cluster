# 2. stop gradient 或者 正负样本匹配
# GCN -> GAT
# feed GCN with edge_attribute
# Peking_1 small-size change data
# network performance hyper-parameter: number of pairs, learning rate, kernel,
# It would be easy to generate a large number of pairs of graphs quickly while human -> manually.
# kernel -> relation graph -> echarts (Peking_1)


from grakel import Graph, ShortestPath
from torch.optim.lr_scheduler import ExponentialLR
from torch_geometric.datasets import ZINC, TUDataset

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import DataLoader
from sklearn import metrics
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import grakel.kernels as ker
from sklearn.preprocessing import normalize

device1 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


class Dataset:
    def __init__(self, name, use_node_attr=False, use_edge_attr=False, device="cpu", load_from_disk=True,
                 kernel=ShortestPath, save_data=False):
        self.device = device
        self.name = name
        self.kernel = kernel
        self.dataset = None
        self.dis = []
        self.K = None
        self.y = []
        # if name in ("ENZYMES", "AIDS", "Mutagenicity", "ZINC","COLLAB","PROTEINS"):
        if name == "ZINC":
            self.dataset = ZINC(root='./', split="test")
        else:
            self.dataset = TUDataset(root='./', name=name, use_node_attr=use_node_attr, use_edge_attr=use_edge_attr)
        if load_from_disk:
            tmp = torch.load(self.name + "_" + str(self.kernel).split(".")[2])
            self.K, self.y, self.dis = tmp['K'], tmp['y'], tmp['dis']
        else:
            self.generate_dis()
            # self.dis = [graph_from_networkx(to_networkx(x)) for x in self.dataset]
            self.calculate_adj_matrix()
            if save_data:
                torch.save({'K': self.K, 'y': self.y, 'dis': self.dis},
                           self.name + "_" + str(self.kernel).split(".")[2])

    def generate_dis(self):
        for data in self.dataset:
            if data.edge_attr is not None:
                edge_attrs = {(int(x), int(y)): tuple(z.numpy().tolist()) for x, y, z in
                              zip(data.edge_index[0], data.edge_index[1], data.edge_attr)}
            else:
                edge_attrs = {(int(x), int(y)): 0 for x, y in zip(data.edge_index[0], data.edge_index[1])}
            edges = set(edge_attrs.keys())
            if data.x is not None:
                node_attributes = {i: tuple(x.numpy().tolist()) for i, x in enumerate(data.x)}
                self.dis.append(Graph(edges, edge_labels=edge_attrs, node_labels=node_attributes, ))
            else:
                self.dis.append(Graph(edges, edge_labels=edge_attrs, node_labels={i: 0 for i in range(data.num_nodes)}))
            if self.name != "ZINC":
                self.y.append(int(data.y))

    def calculate_adj_matrix(self):
        print(self.kernel)
        mat = self.kernel(normalize=True).fit_transform(self.dis)
        # mat = self.kernel(normalize=True).fit_transform(self.dis)
        # mat = normalize(mat,axis=0,norm="max")
        self.K = torch.tensor(mat).to(self.device)


def valid(data):
    if data is None:
        return False
    if data.x is None:
        return False
    if data.x.shape[1] == 0:
        return False
    return True


class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        torch.manual_seed(12345)

        try:
            in_channels = dataset.dataset.num_node_labels
        except Exception:
            in_channels = 1

        self.conv1 = GCNConv(in_channels if in_channels > 0 else 1, 64)
        self.conv1.weight.data.normal_(0.0, 1 / np.sqrt(self.conv1.in_channels))
        self.conv2 = GCNConv(64, 128)
        self.conv2.weight.data.normal_(0.0, 1 / np.sqrt(self.conv2.in_channels))
        self.conv3 = GCNConv(128, 256)
        self.conv3.weight.data.normal_(0.0, 1 / np.sqrt(self.conv3.in_channels))
        # self.lin = Linear(256, 2)
        self.lin = Linear(256, dataset.dataset.num_classes)
        self.lin.weight.data.fill_(0.01)
        # self.lin2 = Linear(512,dataset.dataset.num_classes)
        # self.lin = Linear(512, 200)

    def __forward(self, x1, edge_index1, b1):
        x1 = self.conv1(x1, edge_index1)
        x1 = x1.relu()
        x1 = self.conv2(x1, edge_index1)
        x1 = x1.relu()
        x1 = self.conv3(x1, edge_index1)
        x1 = global_mean_pool(x1, b1)
        # x1 = F.dropout(x1, p=0.5, training=self.training)
        x1 = self.lin(x1)
        # x1 = F.normalize(x1, p=1, dim=-1)
        return x1

    def forward(self, x1, edge_index1, b1, x2=None, edge_index2=None, b2=None, status=0):
        x1 = self.__forward(x1, edge_index1, b1)
        if status == 0:
            x2 = self.__forward(x2, edge_index2, b2)
        return x1, x2


def matching(ans, y):
    dic = {x: [] for x in range(dataset.dataset.num_classes)}
    for ind, x in enumerate(ans):
        dic[x].append(ind)

    dic2 = {x: [] for x in range(dataset.dataset.num_classes)}
    for ind, x in enumerate(y):
        dic2[x].append(ind)

    fin_dic = [[(k1, k2), len(set(v1).intersection(set(v2)))] for (k1, v1) in dic.items() for (k2, v2) in dic2.items()]
    fin_dic.sort(key=lambda x: x[1], reverse=True)

    cc = np.zeros([dataset.dataset.num_classes, dataset.dataset.num_classes])
    for x in fin_dic:
        cc[x[0]] = x[1]

    return cc[linear_sum_assignment(cc, maximize=True)].sum() / len(ans)


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1., pos_margin=0.1, neg_margin=5.):
        super(ContrastiveLoss, self).__init__()
        self.pos_margin = margin if pos_margin is None else pos_margin
        self.neg_margin = margin if neg_margin is None else neg_margin

    def forward(self, x1, x2, label, size_average=True):
        euc_dist = F.pairwise_distance(x1, x2, keepdim=True)
        cont_loss = (label * torch.clamp(euc_dist - self.pos_margin, min=0.0) +
                     (1. - label) * torch.clamp(self.neg_margin - euc_dist, min=0.0))
        return torch.mean(cont_loss)


class ModifiedLoss(torch.nn.Module):
    def __init__(self):
        super(ModifiedLoss, self).__init__()

    def forward(self, x1, x2, in_type, running_type):
        # dis = F.pairwise_distance(x1, x2, keepdim=True)
        dis = F.cosine_similarity(x1, x2)
        # diff = torch.abs(dis - torch.tensor(neg_sim).reshape_as(dis).to(device))
        if running_type:
            diff = torch.abs(dis - (torch.ones_like(dis).to(device1) if in_type else torch.zeros_like(dis).to(device1)))
            diff = torch.mean(diff)
        else:
            diff = torch.sum(dis > 0.99) if in_type else torch.sum(dis < 0.99)
        return diff


dataset = Dataset(name="ZINC", device=device1, load_from_disk=True)


# model.load_state_dict(torch.load("model_Mutagenicity"))


def train():
    model.train()
    loss_val = 0
    for d1, d2 in zip(trainL1, trainL2):
        m1, n1 = d1
        m1, n1 = m1.to(device1), n1.to(device1)
        if not valid(m1):
            m1.x = torch.ones(m1.batch.shape[0], 1).to(device1)
        if not valid(n1):
            n1.x = torch.ones(n1.batch.shape[0], 1).to(device1)

        if m1.x.dtype == torch.int64:
            m1.x = m1.x.float()
        if n1.x.dtype == torch.int64:
            n1.x = n1.x.float()

        x1, x2 = model(m1.x, m1.edge_index, m1.batch, n1.x, n1.edge_index, n1.batch)
        # loss = criterion((x1), (x2),1)
        # loss = criterion((x1), (x2),[1-K[x[0],x[1]] for x in s])
        loss = criterion(x1, x2, True, model.training)
        # dis = F.pairwise_distance(x1, x2, keepdim=True)
        # minus_sim = [1-K[x[0],x[1]] for x in s]
        # loss = torch.mean(torch.tensor([torch.abs(a - b) for a, b in zip(dis, minus_sim)]))
        loss_val += loss
        # loss = - 0.5 * torch.mean(x1 * torch.log(F.softmax(x2)) + x2 * torch.log(F.softmax(x1)))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # for m2, n2, s2 in trainL2:
        m2, n2 = d2
        m2, n2 = m2.to(device1), n2.to(device1)
        if not valid(m2):
            m2.x = torch.ones(m2.batch.shape[0], 1).to(device1)
        if not valid(n2):
            n2.x = torch.ones(n2.batch.shape[0], 1).to(device1)

        if m2.x.dtype == torch.int64:
            m2.x = m2.x.float()
        if n2.x.dtype == torch.int64:
            n2.x = n2.x.float()
        x1, x2 = model(m2.x, m2.edge_index, m2.batch, n2.x, n2.edge_index, n2.batch)
        # loss = criterion((x1), (x2), 0)
        # loss = criterion((x1), (x2),[1-K[x[0],x[1]] for x in s])
        loss = criterion(x1, x2, False, model.training)
        # dis = F.pairwise_distance(x1, x2, keepdim=True)
        # minus_sim = [1-K[x[0],x[1]] for x in s]
        # loss = torch.mean(torch.tensor([torch.abs(a - b) for a, b in zip(dis, minus_sim)]))
        loss_val += loss
        # loss = 0.5 * torch.mean(x1 * torch.log(F.softmax(x2)) + x2 * torch.log(F.softmax(x1)))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return loss_val


@torch.no_grad()
def __val(mm, m, n, rule):
    m, n = m.to(device1), n.to(device1)
    if not valid(m):
        m.x = torch.ones(m.batch.shape[0], 1).to(device1)
    if not valid(n):
        n.x = torch.ones(n.batch.shape[0], 1).to(device1)

    if m.x.dtype == torch.int64:
        m.x = m.x.float()
    if n.x.dtype == torch.int64:
        n.x = n.x.float()
    x1, x2 = mm(m.x, m.edge_index, m.batch, n.x, n.edge_index, n.batch)
    return criterion(x1, x2, rule, mm.training, device1)


@torch.no_grad()
def validate():
    correct_num = 0
    mm = model.to(device1)
    mm.eval()
    for d1, d2 in zip(valL1, valL2):
        m1, n1 = d1
        correct_num += __val(mm, m1, n1, True)
        # m1, n1 = m1.to(device1), n1.to(device1)
        # if not valid(m1):
        #     m1.x = torch.ones(m1.batch.shape[0],1).to(device1)
        # if not valid(n1):
        #     n1.x = torch.ones(n1.batch.shape[0],1).to(device1)
        # x1, x2 = mm(m1.x, m1.edge_index, m1.batch, n1.x, n1.edge_index, n1.batch)
        # correct_num += criterion((x1), (x2),True,mm.training,device1)
        m2, n2 = d2
        correct_num += __val(mm, m2, n2, False)
    return correct_num / 100


@torch.no_grad()
def test(nor=False):
    model.eval()
    fin = []
    for data in test_loader:
        data = data.to(device1)
        if not valid(data):
            data.x = torch.ones(data.batch.shape[0], 1).to(device1)
        if data.x.dtype == torch.int64:
            data.x = data.x.float()
        out = model(data.x, data.edge_index, data.batch, status=1)
        out = out[0].cpu().detach().numpy()
        fin.append(out)
    fin = np.vstack(fin)
    if nor:
        fin = normalize(fin)
    return fin


def triple(fir, sec):
    sec = [x for x in sec]
    tmp = []
    for x in fir:
        for i, y in enumerate(sec):
            if y[0] == x[0] or y[1] == x[0]:
                tmp.append([x[0], x[1], y[1] if y[0] == x[0] else y[0]])
                sec.pop(i)
                if len(tmp) >= 1000:
                    return tmp
                break


aa = torch.tensor(sorted(list({(i, int(y)) if i <= y else (int(y), i) for i, x in enumerate(dataset.K)
                               for y in torch.topk(x, 5)[1] if i != y}), key=lambda x: dataset.K[x[0]][x[1]],
                         reverse=True))
aaa = torch.tensor(sorted(list({(i, int(y)) if i <= y else (int(y), i) for i, x in enumerate(dataset.K)
                                for y in torch.topk(x, 5, largest=False)[1] if i != y}),
                          key=lambda x: dataset.K[x[0]][x[1]], reverse=True))

# print(np.sum([bool(m.y == n.y) for m,n in zip(dataset.dataset[aa[:,0]], dataset.dataset[aa[:,1]])]+[bool(m.y != n.y) for m,n in zip(dataset.dataset[aaa[:,0]], dataset.dataset[aaa[:,1]])]))


# aa = ((dataset.K > 0.9).nonzero())
# aaa = ((dataset.K < 0.1).nonzero())

print(aa.shape[0], np.sum([bool(m.y == n.y) for m, n in zip(dataset.dataset[aa[:, 0]], dataset.dataset[aa[:, 1]])]))
print(aaa.shape[0], np.sum([bool(m.y != n.y) for m, n in zip(dataset.dataset[aaa[:, 0]], dataset.dataset[aaa[:, 1]])]))

# kk = ((dataset.K == 0).nonzero())

# kk = (((0.6 < dataset.K) & (dataset.K < 0.7)).nonzero())

# np.sum([bool(m.y == n.y) for m,n in zip(dataset.dataset[aa[:,0]], dataset.dataset[aa[:,1]])])

# tmp = triple(aa,aaa)

# ll = int(np.min([aa.shape[0],aaa.shape[0]]))
# ll = int(ll*0.01)

ll = 2000

# ll = int(np.min([aa.shape[0],aaa.shape[0]])) - 50

trainL1 = DataLoader(
    (list(zip(dataset.dataset[[x[0] for x in aa[:ll]]], dataset.dataset[[x[1] for x in aa[:ll]]]))),
    batch_size=64, shuffle=True)
trainL2 = DataLoader(
    (list(zip(dataset.dataset[[x[0] for x in aaa[:ll]]], dataset.dataset[[x[1] for x in aaa[:ll]]]))),
    batch_size=64, shuffle=True)

valL1 = DataLoader(
    (list(zip(dataset.dataset[[x[0] for x in aa[ll:ll + 50]]], dataset.dataset[[x[1] for x in aa[ll:ll + 50]]]))),
    batch_size=25, shuffle=True)
valL2 = DataLoader(
    (list(zip(dataset.dataset[[x[0] for x in aaa[ll:ll + 50]]], dataset.dataset[[x[1] for x in aaa[ll:ll + 50]]]))),
    batch_size=25, shuffle=True)

test_loader = DataLoader(dataset.dataset, batch_size=256)

model = GCN(hidden_channels=64).to(device1)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # (SGD Adam)
scheduler1 = ExponentialLR(optimizer, gamma=0.95)
# criterion = ContrastiveLoss().to(device)
criterion = ModifiedLoss().to(device1)
# criterion = MyLoss().to(device)

# cal_d = tmp_reformat(dataset)
# G, y = [x[0] for x in cal_d], [x[1] for x in cal_d]
# K = torch.tensor(ShortestPath(normalize=True).fit_transform(G)).to(device)

# draw_histc(dataset)

loss_list = []
validate_list = []
test_list = []
nmi_list = []
adjusted_rand_score_list = []
sil_list = []
cal_list = []

# train()
for epoch in range(1, 200):
    # t1 = aa[torch.randint_like(torch.zeros(640, dtype=torch.long), 0, len(aa))]
    # t2 = aaa[torch.randint_like(torch.zeros(640, dtype=torch.long), 0, len(aaa))]
    # trainL1 = DataLoader((list(zip(dataset[[x[0] for x in tmp]],dataset[[x[1] for x in tmp]],tmp))), batch_size=64,shuffle=True)
    # trainL2 = DataLoader((list(zip(dataset[[x[0] for x in kk]],dataset[[x[1] for x in kk]],kk))), batch_size=64,shuffle=True)
    # trainL2 = DataLoader((list(zip(dataset[[x[0] for x in tmp]],dataset[[x[2] for x in tmp]],tmp))), batch_size=64,shuffle=True)
    loss_val = train()
    # train_acc = test(test_loader)
    validate_acc = validate()
    res = test()

    kmeans = KMeans(n_clusters=dataset.dataset.num_classes, random_state=0).fit_predict(res)

    nmi_list.append(np.round(metrics.normalized_mutual_info_score(dataset.y, kmeans), 4))
    adjusted_rand_score_list.append(np.round(metrics.adjusted_rand_score(dataset.y, kmeans), 4))
    sil_list.append(np.round(metrics.silhouette_score(res, kmeans), 4))
    cal_list.append(np.round(metrics.calinski_harabasz_score(res, kmeans), 4))

    print(np.bincount(kmeans))

    test_acc = matching(kmeans, dataset.y)

    # if test_acc >= 0.57:
    #     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # (SGD Adam)

    test_list.append(np.round(np.float64((test_acc)), 4))

    # if epoch >= 100:
    #     scheduler1.step()
    # scheduler1.step()

    loss_list.append(np.round(np.float64((loss_val)), 4))
    validate_list.append(np.round(np.float64((validate_acc)), 4))

    print(
        f'Epoch: {epoch:03d}, Loss val:{loss_val:.4f}, Validate Acc: {validate_acc:.4f}, Test Acc: {test_acc:.4f}')
    # print(f'Epoch: {epoch:03d}, Loss val:{loss_val:.4f}, Test Acc: {test_acc:.4f}')

G = dataset.K.cpu().detach().numpy()
fin_km = KMeans(n_clusters=dataset.dataset.num_classes, random_state=0).fit_predict(G)
# print(matching(SpectralClustering(n_clusters=dataset.dataset.num_classes, random_state=0,affinity='precomputed').fit_predict(G),dataset.y))
print(matching(fin_km, dataset.y))
print(metrics.normalized_mutual_info_score(dataset.y, fin_km))

print(loss_list)
print(validate_list)
print(test_list)
print(nmi_list)
print(adjusted_rand_score_list)
print(sil_list)
print(cal_list)
