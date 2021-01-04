import random
from gensim.models import Word2Vec
from graph import *
import pandas as pd
import networkx as nx

name_d = 'flickr'

dataset = 'data/'+name_d+'.mat'
G = load_matfile(dataset)
t = 40  # walk length
r = 80  # number of walks
ns = 5  # negative sample size
w = 10  # window size
worker_threads = 32
embedding_dim = 128
seed = 42

walks = build_deepwalk_corpus(G, num_paths=r, path_length=t, alpha=0, rand=random.Random(seed))

embedder = Word2Vec(walks, window=w, sg=1, hs=1, negative=ns,
                    seed=42, size=embedding_dim, workers=worker_threads, min_count=0)

embedder.wv.save_word2vec_format(name_d+'.embeddings')
