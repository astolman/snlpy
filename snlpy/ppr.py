"""
Implementation for the Andersen-Chung-Lang approximate ppr algorithm

METHODS
-------
ppr(G, seed, alpha=0.85, tol=0.0001)
"""
from numba import njit, prange
from scipy import sparse
import numpy as np
import collections


def ppr(adj, seed, alpha=0.85, tol=0.0001):
    """
    Compute approximate ppr vector for the given seed on the graph

    note: this is stolen from dgleich's github page originally

    PARAMETERS
    ----------
    G : CsrGraph
        The graph on which to perform PPR
    seed : Iterable of ints
        node ids for seeds for PPR walk to teleport to
    alpha : float
        teleportation parameter for PPR
    tol : float
        resolution parameter for PPR, maximum nnz size of result is
        1/tol

    RETURNS
    -------
    scipy.sparse.csr_matrix representation of approximate PPR vector
    """
    p = np.zeros(adj.shape[0])
    r = np.zeros(adj.shape[0])
    Q = collections.deque()  # initialize queue
    r[seed] = 1/len(seed)
    Q.extend(s for s in seed)
    while len(Q) > 0:
        v = Q.popleft()  # v has r[v] > tol*deg(v)
        p, r_prime = push(v, np.copy(r), p, adj.indptr, adj.indices, alpha)
        new_verts = np.where(r_prime - r > 0)[0]
        r = r_prime
        Q.extend(u for u in new_verts if r[u] / np.sum(adj[u].todense()) > tol)
    return sparse.csr_matrix(p)


@njit()
def push(u, r, p, adj_ptrs, adj_cols, alpha):
    r_u = r[u]
    p[u] += alpha * r_u
    r[u] = (1 - alpha) * r_u / 2
    r[adj_cols[adj_ptrs[u]:adj_ptrs[u + 1]]
      ] += (1 - alpha) * r_u / (2 * (adj_ptrs[u + 1] - adj_ptrs[u]))
    return p, r
