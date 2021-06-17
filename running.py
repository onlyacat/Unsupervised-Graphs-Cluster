from read_gxl import getData
import grakel.kernels as gk
from sklearn.cluster import SpectralClustering
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
import numpy as np
import sys

def matching(ans,y):
    dic = {1:[],2:[],3:[],4:[],5:[],0:[]}
    for ind,x in enumerate(ans):
        dic[x].append(ind)

    dic2 = {1:[],2:[],3:[],4:[],5:[],0:[]}
    for ind,x in enumerate(y):
        dic2[x].append(ind)

    fin_dic = [[(k1, k2), len(set(v1).intersection(set(v2)))] for (k1,v1) in dic.items() for (k2, v2) in dic2.items()]
    fin_dic.sort(key=lambda x:x[1],reverse=True)

    cc = np.zeros([6, 6])
    for x in fin_dic:
        cc[x[0]] = x[1]

    return (cc[linear_sum_assignment(cc,maximize=True)].sum()/len(ans))

_,dataset,kernel,cluster = sys.argv


# datasets = [("ENZYMES"),("AIDS"),("MUTA")]
# kernels = [ShortestPath,WeisfeilerLehman,NeighborhoodHash,NeighborhoodSubgraphPairwiseDistance,WeisfeilerLehmanOptimalAssignment]
#
# clustering = ["kmeans","discretize"]

data = getData(dataset)
G = [x[1] for x in data]
y = [x[2] for x in data] if dataset != "ZINC" else None
K = getattr(gk,kernel)(normalize=True).fit_transform(G)
K = np.nan_to_num(K)

num = len(set(y)) if dataset != "ZINC" else 3

sc = SpectralClustering(num, affinity='precomputed', n_init=100,assign_labels=cluster)

ans = sc.fit_predict(K)

res = [["Dataset","Kernel","Cluster_method","Accuracy","silhouette_score","calinski_harabasz_score","davies_bouldin_score"]]

acc = np.Inf if dataset == "ZINC" else np.round(matching(ans,y),3)

res.append([dataset,str(kernel).split('.')[-1].strip(">'"),cluster,acc,
            np.round(metrics.silhouette_score(K, ans, metric='euclidean'),3),
            np.round(metrics.calinski_harabasz_score(K, ans),3),
            np.round(metrics.davies_bouldin_score(K, ans),3)])

row_format ="{:>20}\t" * (len(res[0]))
print(row_format.format(*res[0]))
print(row_format.format(*res[1]))