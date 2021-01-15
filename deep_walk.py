'''
    By Carlos Lago Solas - clago@kth.se
'''
import random
from gensim.models import Word2Vec
from graph import *
import pandas as pd
import networkx as nx
import csv

def get_graph_csv(file):
    G = nx.Graph()
    with open(file, newline='', ) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            G.add_edge(*(row[0], row[1]))
    return G

def to_graph(x):
    G = nx.Graph()
    cx = x.tocoo()
    for i,j,v in zip(cx.row, cx.col, cx.data):
        G.add_edge(*(i, j))
    return G

def lastfm():
    name_d = 'lastfm'
    csv_network = 'data/lastfm_asia_edges.csv'
    G = get_graph_csv(csv_network)
    return G, name_d

def blogcatalog():
    name_d = 'blogcatalog'
    dataset = 'data/'+name_d+'.mat'
    # G = load_matfile(dataset)
    mat = loadmat(dataset)
    A = mat['network']
    G = to_graph(A)
    return G, name_d

def pos():
    name_d = 'POS'
    dataset = 'data/'+name_d+'.mat'
    # G = load_matfile(dataset)
    mat = loadmat(dataset)
    A = mat['network']
    G = to_graph(A)
    return G, name_d

def sapiens():
    name_d = 'Homo_sapiens'
    dataset = 'data/'+name_d+'.mat'
    G = load_matfile(dataset)
    return G, name_d

def flickr():
    name_d = 'flickr'
    dataset = 'data/'+name_d+'.mat'
    G = load_matfile(dataset)
    return G, name_d

def pubmed():
    name_d = 'pubmed'
    csv_network = 'data/pubmed/pubmed_edges.csv'
    G = get_graph_csv(csv_network)
    return G, name_d

def cora():
    name_d = 'cora'
    csv_network = 'data/cora/cora_edges.csv'
    G = get_graph_csv(csv_network)
    return G, name_d


def get_random_walk(graph, node, steps):
    path = [str(node), ]
    next_node = node
    for _ in range(steps):
        neighbors = list(nx.all_neighbors(graph, next_node))
        next_node = random.choice(neighbors)
        path.append(str(next_node))
    return path


G, name_d = pubmed()


t = 40  # walk length
r = 80  # number of walks
ns = 5  # negative sample size
w = 10  # window size
worker_threads = 32
embedding_dim = 128
seed = 42

walks = []
for node in G.nodes():
    for _ in range(r):
        walks.append(get_random_walk(G, node, t-1))

embedder = Word2Vec(walks, window=w, sg=1, hs=1, negative=ns,
                    seed=42, size=embedding_dim, workers=worker_threads, min_count=0)

embedder.wv.save_word2vec_format(name_d+'.embeddings')
