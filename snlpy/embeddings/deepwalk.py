"""
Implementation for DeepWalk
"""
from numba import njit, prange
from numpy import random
import numpy as np
from gensim.models.word2vec import Word2Vec


def DeepWalk(adj, walk_number=10, walk_length=80, dimensions=128,
             workers=4, window_size=5, epochs=1, learning_rate=0.05):
    """
    Fit an embedding to graph according to DeepWalk method

    PARAMETERS
    ----------
    graph : scipy.sparse.csr_matrix
        adjacency matrix of graph to which to fit an embedding
    walk_number : int, optionl
        number of walks for DeepWalk
    walk_length : int, optionl
        length of walks for DeepWalk
    dimensions : int, optionl
        number of dimensions for the embedding
    workers : int, optionl
        number of workers for the Word2Vec step
        (random walks use all available cores)
    window_size : int, optionl
        window size for Word2Vec
    epochs : int, optionl
        number of iterations for Word2Vec
    learning_rate : float, optionl
        parameter for Word2Vec

    RETURNS
    -------
    np.ndarray(shape=(adj.shape[0], d), dtype=np.float32)
    """
    walk_container = _do_walks(adj, walk_length, walk_number)
    model = Word2Vec(walk_container,
                     hs=1,
                     alpha=learning_rate,
                     iter=epochs,
                     size=dimensions,
                     window=window_size,
                     min_count=1,
                     workers=workers,
                     )
    emb = np.zeros((adj.shape[0], dimensions))
    for i in range(adj.shape[0]):
        emb[i, :] = model[str(i)]
    return emb


@njit()
def do_step(v, rows, indxs):
    """does one step of a walk from v

    PARAMETERS
    ----------
    v : int
        vertex from which to step
    rows : np.ndarray
        array containing all rows of adjacency matrix concatenated
    indxs : np.ndarray
        array of pointers into rows

    RETURNS
    _______
    int
        next step in random walk
    """
    return random.choice(rows[indxs[v]:indxs[v + 1]])


@njit(parallel=True)
def do_walk(rows, indxs, num_steps, endpoints, walks):
    """
    does a walk from every vertex given in endpoints

    PARAMETERS
    ----------
    rows : np.ndarray
        array containing column indices of all nonzero coordinates
        in the adjacency matrix
    indxs : np.ndarray
        array of pointers into rows indicating the start of each row
    num_steps : int
        length of walk to perform
    endpoints : np.ndarray
        array of endpoints from which to start walks
    walks : np.ndarray
        empty placeholder array which will be filled with walk transcripts

    RETURNS
    _______
    np.ndarray containing walk transcripts
    """
    walks[:, 0] = endpoints
    for v in prange(len(endpoints)):
        for j in range(1, num_steps):
            walks[v, j] = do_step(walks[v, j - 1], rows, indxs)
    return walks


def process_adj(g):
    rows = g.indices
    indxs = g.indptr
    return rows, indxs


class _WalkContainer():
    """Iterator containing the walk transcripts"""

    def __init__(self, height, width):
        """height, width: ints for size of walk container"""
        self.walks = np.zeros((height, width), dtype=np.int32)

    def __iter__(self):
        for walk in self.walks:
            yield [str(x) for x in walk]


def _do_walks(adj, walk_length, walk_number):
    """
    Perform random walks


    PARAMETERS
    ----------
    adj : scipy.sparse.csr_matrix
        adjacency matrix of graph to do walks on
    walk_length : int
        length of random walks
    walk_number : int
        number of random walks per vertex

    RETURNS
    -------
    iterator containing walk as lists of strings
    """
    rows, indxs = process_adj(adj)
    n = len(indxs) - 1
    walk_container = _WalkContainer(n * walk_number, walk_length)
    for i in range(walk_number):
        endpoints = np.arange(n, dtype=np.int32)
        do_walk(rows, indxs, walk_length, endpoints,
                walk_container.walks[i * n:(i + 1) * n])
    return walk_container
