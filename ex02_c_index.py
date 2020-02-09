import numpy as np

def c_index(X, labels):
    n = X.shape[0]
    distance = np.zeros((n, n))
    for j in range(n):
        distance[j, :] = np.sqrt(((X-X[j, :])**2).sum(axis=1))

    clust_con = np.zeros((n, n)) # Matrix which contains the information, whether a sample is in the same cluster or not
    for j in range(len(labels)):
        clust_con[j, :] = (labels[j] == labels)
        clust_con[j, j] = 0 # set diagonal matrix to 0

    S_cl = (distance * clust_con).sum()/2
    q = int(clust_con.sum()/2)

    distance_ranked = np.sort(distance[np.triu_indices(n, k=1)]) # distance from all samples to each other

    S_min = distance_ranked[:q].sum()
    S_max = distance_ranked[-q:].sum()

    c = (S_cl - S_min)/(S_max - S_min)
    return c