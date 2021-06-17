# Unsupervised-Graphs-Cluster

---
### Introduction
In the report, we used one pipeline to deal with the unsupervised graph clustering method, which contain two parts: 
- one is the computation of the adjacent matrix, 
- the other is the unsupervised clustering methods based only on the adjacent matrix. 
  
We evaluate our model on the challenging problem of graphs clustering that plays an important role in the classification of molecules in chemical compounds systems. 

The experimental analysis demonstrates that our models are not only able to deal with the labeled datasets but they can also give an reasonable clustering result on the unlabelled dataset.

---
### How to run

1. Requirements
- Python    3.+
- sklearn   0.24.1
- scipy     1.6.2
- numpy     1.19
- grakel    0.1.8

2. Command lines
```
python3 ./running.py [dataset] [kernel] [cluster_method]
```
where,

- dataset : 
    - `ENZYMES`
    - `AIDS`
    - `MUTA`
    - `ZINC`

- kernel : 
  - `ShortestPath`
  - `WeisfeilerLehman`
  - `NeighborhoodHash`
  - `NeighborhoodSubgraphPairwiseDistance`
  - `WeisfeilerLehmanOptimalAssignment`

- cluster_method 
  -  `kmeans`
  - `discretize`


3. Results
```
$ python3 ./running.py ENZYMES ShortestPath discretize
Dataset                  Kernel          Cluster_method                Accuracy        silhouette_score    calinski_harabasz_score davies_bouldin_score    
ENZYMES            ShortestPath              discretize                   0.258                   0.161                 139.304                   1.493    
```

```
$ python3 ./running.py ZINC ShortestPath discretize
Dataset                  Kernel          Cluster_method                Accuracy        silhouette_score    calinski_harabasz_score davies_bouldin_score    
   ZINC            ShortestPath              discretize                     inf                   0.163                1467.447                   2.427 
```