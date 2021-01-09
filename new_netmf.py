import scipy.io
import scipy.sparse as sparse
from scipy.sparse import csgraph
import numpy as np
import argparse
import logging
import theano
from theano import tensor as T

# file = 'blogcatalog'
# file = 'flickr'
file = 'PubMed'
# file = 'POS'
# file = 'Homo_sapiens'

b = negative = 1
window = 10
dim = 128

# read the file
# data = scipy.io.loadmat(file+'.mat')
# A = data['network']
data = open(file+'.edges').readlines()
data = np.array([[int(dd) for dd in d.strip().split(',')] for d in data])

row = data[:,0]
col = data[:,1]

vertices = set(np.concatenate((row,col),axis=0))

map_vertex = {vertex:i for i,vertex in enumerate(vertices)}

row = np.array([map_vertex[v] for v in row])
col = np.array([map_vertex[v] for v in col])

shape = (len(map_vertex),len(map_vertex))

data = np.array([1]*len(row))

A = sparse.csr_matrix((data,(row,col)), shape=shape)
A = A+A.transpose()
A[A>0] = 1

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

# print(check_symmetric(A.todense()))

# write the function
vol = float(A.sum())
n = A.shape[0]
L, d_rt = csgraph.laplacian(A, normed=True, return_diag=True)
X = sparse.identity(n) - L

if file=='flickr': rank = 16384
else: rank = 256
evals, evecs = sparse.linalg.eigsh(X, rank, which="LA")
D_rt_inv = sparse.diags(d_rt ** -1)
D_rt_invU = D_rt_inv.dot(evecs)

# print(evals, D_rt_invU)

def deepwalk_filter(evals, window):
    for i in range(len(evals)):
        x = evals[i]
        evals[i] = 1. if x >= 1 else x*(1-x**window) / (1-x) / window
    evals = np.maximum(evals, 0)
    return evals
evals = deepwalk_filter(evals, window=window)
# print(evals)
X = sparse.diags(np.sqrt(evals)).dot(D_rt_invU.T).T
m = T.matrix()
mmT = T.dot(m, m.T) * (vol/b)
f = theano.function([m], T.log(T.maximum(mmT, 1)))
Y = f(X.astype(theano.config.floatX))
deepwalk_matrix = sparse.csr_matrix(Y)

u, s, v = sparse.linalg.svds(deepwalk_matrix, dim, return_singular_vectors="u")
deepwalk_embedding = sparse.diags(np.sqrt(s)).dot(u.T).T

np.save("netmf_pubmed.npy", deepwalk_embedding, allow_pickle=False)
