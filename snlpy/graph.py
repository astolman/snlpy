"""module for the CsrDict data structure"""
import numpy as np
from collections.abc import Mapping
from scipy.sparse import csr_matrix
from emBench.graphs.csrDict import CsrDict
import numpy as np
import pickle as pkl


class CsrGraph(CsrDict):
    """Class to provide efficient *immutable* maps for sparse, unweighted graphs.
    exposes a dictlike interface, but errors out on any attempt to set items

    CSR stands for "compressed sparse row" - see the wikipedia link for details.
    Here, we assume all the entries are just ones, and do not need the data array.
    https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)

    ATTRIBUTES
    ----------
    row_indxs : numpy.ndarray
        array containing the column ids for all the values stored
    row_ptrs : nump.ndarray
        array containing the pointers for new row locations into row_indxs

    METHODS
    -------
    degree(node)
        Returns the degree of node.
    eges()
        Returns an iterator over edges in graph.
    keys()
        Returns all nodes in the graph (same as nodes()).
    neighbors(v)
        Returns the neighbors of v.
    nodes()
        Returns an iterator over the nodes in the graph.
    number_of_edges()
        Returns the number of edges in the graph.
    number_of_nodes()
        Returns the number of nodes in the graph.
    remove_isolates()
        Returns a CsrGraph with all isolated nodes removed from the graph and
        relabels the remaining sequentially.
    save(filename)
        Write this object to python pickle file given by filename.
    to_csr_matrix()
        Returns adjacency matrix as scipy.sparse.csr_matrix.
    """

    def __init__(self, adjList, symmetric=False):
        """
        PARAMETERS
        ----------
        adjList : dictlike
            an adjacency list representation of the graph in a dictlike object keyed by consecutive node id and
            with values as iterable over node ids. Node ids must be consecutive ints starting from 0.
            Only needs to support [] indexing
        """
        self.row_indxs = np.empty(0, dtype=np.int32)
        self.row_ptrs = np.zeros(1, dtype=np.int32)
        index = 0
        for v in range(len(adjList)):
            row = sorted(adjList[v])
            self.row_indxs = np.concatenate((self.row_indxs, row))
            index += len(row)
            self.row_ptrs = np.concatenate((self.row_ptrs, [index]))

    def number_of_nodes(self):
        """Returns the number of nodes

        RETURNS
        -------
        int"""
        return len(self)

    def nodes(self):
        """Returns iterator over nodes in graph (alias for self.keys())"""
        return self.keys()

    def neighbors(self, v):
        """Returns the neighbors of node v

        PARAMETERS
        ----------
        v: int
            index of node

        RETURNS
        -------
        numpy.ndarray containing indices of neighbors of v
        """
        return self[v]

    def degree(self, node):
        """Returns degree of node in graph

        PARAMETERS
        ----------
        node : int
            a node in the graph

        RETURNS
        -------
        int representing the size of node's neighborhood
        """
        return len(self[node])

    def number_of_edges(self):
        """Returns number of edges in graph"""
        return len(self.row_indxs) // 2

    def edges(self):
        """Returns iterator over all edges in graph"""
        edges = np.zeros([self.number_of_edges(), 2], dtype=np.int32)
        ind = 0
        for v in self.nodes():
            for u in self.neighbors(v)[self.neighbors(v) < v]:
                edges[ind, 0] = v
                edges[ind, 1] = u
                ind += 1
        return edges

    def remove_isolates(self):
        """Return value: SparseGraphWC that is isomorphic to self but has no zero degree nodes"""
        isolates = [v for v in self.nodes() if len(self.neighbors(v)) == 0]
        relabel_map = {v: v - len([y for y in isolates if y < v])
                       for v in self.nodes() if v not in isolates}
        adj = CsrDict({relabel_map[v]: [relabel_map[u] for u in self.neighbors(
            v)] for v in self.nodes() if v not in isolates})
        return CsrGraph(adj), relabel_map

    def save(self, filename):
        """save object to filename with pickle

        PARAMETERS
        ----------
        filename : string
            path which to save pickle file to
        """
        filep = open(filename, "wb")
        pkl.dump(self, filep)
        filep.close()

    def __getitem__(self, key):
        """Returns the array of values corresponding to key

        PARAMETERS
        ----------
        key : int
            a key in the dict

        RETURNS
        -------
        np.ndarray of values
        """
        # TODO should I be raising a value error?
        if key == len(self.row_ptrs) - 1:
            return self.row_indxs[self.row_ptrs[key]:]
        elif key >= len(self.row_ptrs):
            return []
        return self.row_indxs[self.row_ptrs[key]:self.row_ptrs[key + 1]]

    def keys(self):
        """Returns the keys in this object

        RETURNS
        -------
        np.ndarray(dtype=np.intew)
        """
        return np.arange(len(self), dtype=np.int32)

    def __len__(self):
        return len(self.row_ptrs) - 1

    def __iter__(self):
        for i in range(len(self)):
            yield i

    def to_csr_matrix(self):
        """Convert self to scipy.csr_matrix

        RETURNS
        _______
        scipy.sparse.csr_matrix
        """
        return csr_matrix((np.ones(len(self.row_indxs), dtype=np.bool_),
                           self.row_indxs, self.row_ptrs))

    def transpose(self):
        """Returns a CsrDict keyed by values of self and with values equal
        to the set of keys in self which map to a given index

        RETURNS
        -------
        CsrDict
        """
        a = self.to_csr_matrix()
        b = a.tocsc()
        c = CsrDict(dict())
        c.row_ptrs = b.indptr
        c.row_indxs = b.indices
        return c
