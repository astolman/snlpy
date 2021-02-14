"""
Implementation of NetMF embedding method
"""
from scipy import sparse
import numpy as np
from sklearn.decomposition import TruncatedSVD



def NetMF(graph, dimensions=128, iterations=10, order=2, negative_samples=1):
    """
    Fits a NetMF embedding to the given graph.

    PARAMETERS
    ----------
    graph : CsrGraph
        Graph to which to fit an embedding
    dimensions : int, optional
        Number of dimensions for embedding, default is 128
    iterations : int, optional
        Number of iterations to run NetMF
    order : int, optional
        Power of matrix to go up to for NetMF
    negative_samples : int, optional
        Parameter for NetMF

    RETURNS
    -------
    np.ndarray(shape=(graph.number_of_nodes(), d), dtype=np.float32)
    """
    target_matrix = _create_target_matrix(graph, order, negative_samples)
    return _create_embedding(target_matrix, dimensions, iterations)

def _create_D_inverse(graph):
    """
    Creating a sparse inverse degree matrix.

    Arg types:
        * **graph** *(NetworkX graph)* - The graph to be embedded.

    Return types:
        * **D_inverse** *(Scipy array)* - Diagonal inverse degree matrix.
    """
    index = np.arange(graph.number_of_nodes())
    values = np.array([1.0/graph.degree(node) for node in range(graph.number_of_nodes())])
    shape = (graph.number_of_nodes(), graph.number_of_nodes())
    D_inverse = sparse.csr_matrix((values, (index, index)), shape=shape, dtype=np.float32)
    return D_inverse

def _create_base_matrix(graph):
    """
    Creating the normalized adjacency matrix.

    Arg types:
        * **graph** *(NetworkX graph)* - The graph to be embedded.

    Return types:
        * **(A_hat, A_hat, A_hat, D_inverse)** *(SciPy arrays)* - Normalized adjacency matrices.
    """
    A = graph.to_csr_matrix()
    D_inverse = _create_D_inverse(graph)
    A_hat = D_inverse.dot(A)
    return (A_hat, A_hat, A_hat, D_inverse)

def _create_target_matrix(graph, order, negative_samples):
    """
    Creating a log transformed target matrix.

    Arg types:
        * **graph** *(NetworkX graph)* - The graph to be embedded.

    Return types:
        * **target_matrix** *(SciPy array)* - The shifted PMI matrix.
    """
    A_pool, A_tilde, A_hat, D_inverse = _create_base_matrix(graph)
    for _ in range(order-1):
        A_tilde = A_tilde.dot(A_hat)
        A_pool = A_pool + A_tilde
    del A_hat, A_tilde
    A_pool.data = (graph.number_of_edges()*A_pool.data)/(order*negative_samples)
    A_pool = A_pool.dot(D_inverse)
    A_pool.data[A_pool.data < 1.0] = 1.0
    target_matrix = sparse.csr_matrix((np.log(A_pool.data), A_pool.indices, A_pool.indptr),
                                      shape=A_pool.shape,
                                      dtype=np.float32)
    return target_matrix

def _create_embedding(target_matrix, dimensions, iterations):
    """
    Fitting a truncated SVD embedding of a PMI matrix.
    """
    svd = TruncatedSVD(n_components=dimensions,
                       n_iter=iterations)
    svd.fit(target_matrix)
    embedding = svd.transform(target_matrix)
    return embedding
