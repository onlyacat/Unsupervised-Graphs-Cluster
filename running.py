from read_gxl import getData
from grakel.kernels import ShortestPath,WeisfeilerLehman,NeighborhoodHash,NeighborhoodSubgraphPairwiseDistance,WeisfeilerLehmanOptimalAssignment
from sklearn.cluster import SpectralClustering
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
import numpy as np

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

datasets = [("ENZYMES"),("AIDS"),("MUTA")]
kernels = [ShortestPath,WeisfeilerLehman,NeighborhoodHash,NeighborhoodSubgraphPairwiseDistance,WeisfeilerLehmanOptimalAssignment]

clustering = ["kmeans","discretize"]
res = []

for dataset in datasets:
    data = getData(dataset)
    G, y = [x[1] for x in data], [x[2] for x in data]
    for kernel in kernels:
        K = kernel(normalize=True).fit_transform(G)
        K = np.nan_to_num(K)
        for cluster in clustering:
            sc = SpectralClustering(len(set(y)), affinity='precomputed', n_init=100,assign_labels=cluster)

            ans = sc.fit_predict(K)
            res.append([dataset,str(kernel).split('.')[-1].strip(">'"),cluster,matching(ans,y),
                        metrics.silhouette_score(K, ans, metric='euclidean'),
                        metrics.calinski_harabasz_score(K, ans),
                        metrics.davies_bouldin_score(K, ans)])

print(res)