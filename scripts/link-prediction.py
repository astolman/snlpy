import getopt
from sys import argv
import numpy as np
import matplotlib.pyplot as plt
from snlpy import structural_feats

import random

from sklearn.model_selection import ShuffleSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

import argparse

parser = argparse.ArgumentParser(description='compute link prediction stuff')
parser.add_argument("--loadmodels", "-l", help="load models instead of training them",
                    action="store_true")
parser.add_argument("--no_roc", "-n", help="don't compute roc scores",
                    action="store_true")
parser.add_argument("--no_struct", "-s", help="don't compute roc scores",
                    action="store_true")
parser.add_argument("data_path", help="path to directory containing dataset")
parser.add_argument("num_rows", help="number of rows for reliability curve", type=int)
args = parser.parse_args()


def sample_rows(graph, num_rows):
    """
    Provide coordinates for all entries in num_rows number of random rows

    :param graph: CsrGraph to choose rows from.
    :param num_rows: Number of rows to sample.
    """
    rows = np.random.randint(0, high=graph.number_of_nodes(), size=num_rows, dtype=np.int32)
    pairs = np.vstack([np.array([u, v]) for u in graph.nodes() for v in rows])
    return pairs

def histofy(ps, num_bins=1000):
    bins = np.zeros(num_bins, dtype=np.float32)
    for p in ps:
        bins[int( np.floor( p * (num_bins - 1)) )] += 1
    accumulator = 0
    for i in range(num_bins):
        accumulator += bins[-(i + 1)]
        bins[-(i + 1)] = accumulator / len(ps)
    print(bins)
    return bins

##

data_path = args.data_path

##
embedding_names = ['deepwalk', 'netmf', 'node2vec', 'structural']
if args.no_struct:
    embedding_names = embedding_names[:-1]
#embedding_names = ['netmf']
import pickle as pkl
adj = pkl.load(open(data_path + '/adj.pickle', 'rb'))
label_counts = np.asarray(np.sum(adj, axis=1)).reshape(-1)
shuffle = ShuffleSplit(n_splits=1, test_size=0.5)
train_ind, test_ind = list( shuffle.split(range(adj.nnz)) )[0]
train_pair = np.vstack(adj.nonzero()).T[train_ind]
test_pair = np.vstack(adj.nonzero()).T[test_ind]
rng = np.random.default_rng()
neg_train = rng.integers(low=0, high=adj.shape[0], size=(len(train_ind), 2))
train_pair = np.vstack((train_pair, neg_train))
y_train = np.zeros( (2*len(train_ind), 1), dtype=bool)
y_train[:len(train_ind)] = True
neg_test = rng.integers(low=0, high=adj.shape[0], size=(len(test_ind), 2))
test_pair = np.vstack((test_pair, neg_test))
y_test= np.zeros( (2*len(test_ind), 1), dtype=bool)
y_test[:len(test_ind)] = True

fig, ax = plt.subplots()
num_rows = int(argv[2])
sample_verts = rng.integers(low=0, high=adj.shape[0], size=num_rows)
#sample_verts = random.choices(range(adj.shape[0]), weights=label_counts, k=num_rows)
print(label_counts[sample_verts])
fig, ax = plt.subplots()
for emb in embedding_names:
    print(emb)
    if emb == 'structural':
        ppr = pkl.load(open('%s/structural.pickle' % (data_path), 'rb'))
        if not args.loadmodels:
            X = structural_feats(np.vstack((train_pair, test_pair)), adj, ppr)
            X_train = X[:len(train_pair)]
        if not args.no_roc:
            X_test = X[len(train_pair):]
    else:
        vecs = np.load('%s/%s.npy' % (data_path, emb))
        X = np.multiply(vecs, vecs)
        if not args.loadmodels:
            X_train = np.multiply(vecs[train_pair[:, 0]], vecs[train_pair[:, 1]])
        if not args.no_roc:
            X_test = np.multiply(vecs[test_pair[:, 0]], vecs[test_pair[:, 1]])
    if not args.loadmodels:
        clf = OneVsRestClassifier(
                LogisticRegression(
                    solver="liblinear",
                    multi_class="ovr"))
        print('Training model')
        clf.fit(X_train, y_train)
        print('done training')
        pkl.dump(clf, open('%s/%s-lp-model.pickle' % (data_path, emb), 'wb'))
    else:
        clf = pkl.load(open('%s/%s-lp-model.pickle' % (data_path, emb), 'rb'))
        print('model loaded')
    if not args.no_roc:
        test_scores = clf.predict_proba(X_test)[:, 1]
        print("ROC AUC SCORE: %f" % roc_auc_score(y_test, test_scores))
        p, r, t = precision_recall_curve(y_test, test_scores)
        print("PR AUC SCORE: %f" % auc(r, p))


    ps = []
    for v in sample_verts:
        pairs = np.array([[v, i] for i in range(adj.shape[0])], dtype=np.int32)
        if emb != 'structural':
            feats = np.multiply(X[v], X)
        else:
            feats = structural_feats(pairs, adj, ppr)
        row_scores = clf.predict_proba(feats)[:, 1]
        precision = np.sum([adj[pair[0], pair[1]] for pair in pairs[np.argsort(-row_scores)[:label_counts[v]]]]) / label_counts[v]
        ps.append(precision)

    ax.plot(np.arange(1000) / 1000, histofy(ps), label=emb)

graph_name = data_path.split('/')[-2]
ax.set_title('%s link prediction reliability curve' % (graph_name))
ax.set_ylabel('Percent of nodes')
ax.set_xlabel('Precision@d')

ax.legend()
fig.savefig('figures/%s-lp-reliability-curve.png' % (graph_name))
