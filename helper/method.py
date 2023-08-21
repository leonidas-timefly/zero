import faiss
import numpy as np
import time
from sklearn.cluster import AgglomerativeClustering


def Naive_Clustering(data, cluster_number):
    data_num, length = data.shape
    # faiss implementation of k-means
    clus = faiss.Clustering(length, cluster_number)
    # Change faiss seed at each k-means so that the randomly picked
    # initialization centroids do not correspond to the same feature ids
    # from an epoch to another.
    clus.seed = np.random.randint(1234)
    clus.niter = 20
    clus.max_points_per_centroid = 10000000

    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, length, flat_config)

    # perform the training
    print('==> Start clustering..')
    start_time = time.time()
    clus.train(data, index)
    _, I = index.search(data, 1)
    end_time = time.time()
    print('==> Clustering time: {}s'.format(end_time - start_time))
    print('==> Clustering finished..')

    return [int(n[0]) for n in I]


def Hierarchical_Clustering(data):
