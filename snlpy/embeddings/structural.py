import numpy as np
from numba import njit, prange

@njit(parallel=True)
def compute_features(X, community, j, row_ptrs, row_indxs):
    for i in prange(len(row_ptrs) - 1):
        for u in row_indxs[row_ptrs[i]:row_ptrs[i + 1]]:
            for c in community:
                if u == c:
                    X[i, j] += 1
                    break
    return X


def Structural(adj, communities):
    """
    Fit structural embedding

    PARAMETERS
    ----------
    adj : scipy.sparse.csr_matrix
        adjacency matrix of graph
    communities : np.ndarray(shape = (n, k))
        matrix of community identities

    RETURNS
    -------
    np.ndarray(shape=(adj.shape[0], d), dtype=np.int32)
    """

    X = np.zeros(communities.shape)
    for i, comm in enumerate(communities):
        X = compute_features(X, comm, i, adj.indptr, adj.indices)
    return X
