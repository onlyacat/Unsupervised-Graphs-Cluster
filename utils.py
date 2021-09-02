import matplotlib.pyplot as plt
import torch
import numpy as np

def valid(data):
    if data is None:
        return False
    if data.x is None:
        return False
    if data.x.shape[1] == 0:
        return False
    return True

def generate_histc(dataset):
    ini_val = np.float64(0.0)
    num_pairs = []
    pos = []
    neg = []
    while ini_val<=1.0:
        kk = ((((ini_val <= dataset.K) & (dataset.K < ini_val+0.01)) if ini_val < 1.0 else (dataset.K==1.0)).nonzero())
        total_nums = kk.shape[0]
        pos_num = np.sum([bool(m.y == n.y) for m,n in zip(dataset.dataset[kk[:,0]], dataset.dataset[kk[:,1]])])
        pos.append(np.round(pos_num/total_nums,3))
        neg.append(np.round((total_nums - pos_num)/total_nums,3))
        num_pairs.append(total_nums)
        ini_val = np.round(ini_val+ np.float64(0.01),3)
    num_pairs_percentage = np.array(num_pairs) / (dataset.K.shape[0] * dataset.K.shape[0]) * 100

    return num_pairs_percentage,pos,neg


def draw_histc(dataset):
    npp,pos,neg = generate_histc(dataset)

    X = np.linspace(0,1,101,endpoint=True)

    plt.figure(figsize=(10.24,4.8))

    a1 = plt.subplot(1, 2, 1)
    a1.plot(X,npp,color="blue",linewidth=2.5,linestyle="-",label="% Number of graph pairs")
    a1.set_xticks(np.arange(0,1.1,0.1))
    a1.legend(loc='upper right')
    a1.set_xlabel("Pairs Similarity")
    a1.set_ylabel("% of numbers of pairs")

    a2 = plt.subplot(1, 2, 2)
    a2.set_xticks(np.arange(0, 1.1, 0.1))
    a2.plot(X,pos,color="red",linewidth=2.5,linestyle="-",label="Matching Pairs")
    a2.plot(X,neg,color="green",linewidth=2.5,linestyle="-",label="Un-matching Pairs")
    a2.legend(loc='upper left')
    a2.set_xlabel("Pairs Similarity")
    a2.set_ylabel("% of numbers of pairs")

    plt.show()

    print(1)
