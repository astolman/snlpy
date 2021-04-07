import numpy as np
from multiprocessing import Pool
from scipy import sparse

from numba import njit, prange

def structural_feats(pairs, adj, ppr, chunksize=10000):
    chunked_pairs = [pairs[i * chunksize:(i + 1) * chunksize]
                     for i in range(int(len(pairs) / chunksize))]
    if int(len(pairs) / chunksize) * chunksize != len(pairs):
        chunked_pairs += [pairs[int(len(pairs) / chunksize) * chunksize:]]
    return np.vstack([_batch_features(batch, adj, ppr) for batch in chunked_pairs])

def _batch_features(batch, adj, ppr):
    if len(batch) == 0:
        return
    cosines = cosine_sim(batch, adj)
    connectivities = compute_three_paths(batch, adj)
    pprs = np.array(list(map(
            lambda x: [ppr[x[0], x[1]], ppr[x[1], x[0]]], batch)))
    return np.hstack((cosines, connectivities, pprs))

def compute_three_paths(pairs, adj):
    """
    Computes number of paths of legth 3 between vertices in pairs.

    PARAMETERS
    ----------
    pairs : np.ndarray(dtype=np.int32, shape=(k, 2))
        array of pairs
    graph : CsrGraph
        graph to use

    RETURNS
    -------
    np.ndarray(dtype=np.int32, shape=(k,1))
        array of number of paths of length 3
    """
    row_ptrs = adj.indptr
    col_indxs = adj.indices
    connectivities = np.zeros((len(pairs), 1))
    u_neighb = adj[pairs[:, 1]].todense()
    return _compute_three_paths_aggregate(connectivities, pairs[:, 0],
                                         row_ptrs, col_indxs,
                                         u_neighb)

@njit(parallel=True)
def _compute_three_paths_aggregate(feature_vec, vs, row_ptrs, col_indxs, u_neighb):
    for i in prange(len(vs)):
        v = vs[i]
        for k in col_indxs[row_ptrs[v] : row_ptrs[v + 1]]:
            for l in col_indxs[row_ptrs[k]:row_ptrs[k + 1]]:
                if u_neighb[i, l]:
                    feature_vec[i] += 1
    return feature_vec

def cosine_sim(pairs, m):
    """
    Computes cosine similarities between all pairs of
    vertices in pairs.

    PARAMETERS
    ----------
    pairs : np.ndarray(dtype=np.int32, shape=(k, 2))
        array of pairs to compute cosine similarities for
    m : scipy.sparse.csr_matrix
        csr_matrix representation of graph to use to
        compute cosine similarities

    RETURNS
    -------
    np.ndarray(dtype=np.float32, shape=(k,1))
        array of cosine similarities
    """
    left_adj_vectors = m[pairs[:, 0]]
    right_adj_vectors = m[pairs[:, 1]]
    lav_ptr = left_adj_vectors.indptr
    lav_col = left_adj_vectors.indices
    rav_ptr = right_adj_vectors.indptr
    rav_col = right_adj_vectors.indices
    cosines = np.zeros(len(rav_ptr) - 1)
    return _cosines(lav_ptr, lav_col, rav_ptr, rav_col, cosines).reshape(-1, 1)

@njit(parallel=True)
def _cosines(lav_ptr, lav_col, rav_ptr, rav_col, cosines):
    for i in prange(len(cosines)):
        cosines[i] = _cosine_sim_pair(lav_col[lav_ptr[i]:lav_ptr[i + 1]],
                                      rav_col[rav_ptr[i]:rav_ptr[i + 1]])
    return cosines

@njit()
def _cosine_sim_pair(left_ind, right_ind):
    if len(left_ind) == 0 or len(right_ind) == 0:
        return 0.0
    factor = 1 / np.sqrt(len(left_ind) * len(right_ind))
    cosine = 0
    i = 0
    j = 0
    while i < len(left_ind) and j < len(right_ind):
        if left_ind[i] == right_ind[j]:
            cosine += 1
            i += 1
            j += 1
        elif left_ind[i] < right_ind[j]:
            i += 1
        else:
            j += 1
    return factor * cosine
