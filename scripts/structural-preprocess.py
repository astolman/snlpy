from snlpy.ppr import ppr
from scipy import sparse
from sys import argv
import numpy as np
import pickle as pkl

def ppr_prepocess(adj, alpha=0.8, tol=0.00001, workers=20, seeds=None):
    if seeds is None:
        seeds = range(adj.shape[0])
    return sparse.vstack([ppr(adj, [seed], alpha, tol) for seed in seeds])


data_path = argv[1]
adj = pkl.load(open(data_path + '/adj.pickle', 'rb'))
ppr = ppr_prepocess(adj)
pkl.dump(ppr, open('%s/structural.pickle' % (data_path), 'wb'))
