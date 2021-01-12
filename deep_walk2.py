import random
from gensim.models import Word2Vec
from graph import *
import pandas as pd
import networkx as nx
import csv


def get_graph_csv(file, undirected=True):
    G = Graph()

    with open(file, newline='', ) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            G[row[0]].append(row[1])

    if undirected:
        G.make_undirected()

    G.make_consistent()
    return G

def lastfm():
    name_d = 'lastfm'
    csv_network = 'data/lastfm_asia_edges.csv'
    G = get_graph_csv(csv_network)
    return G, name_d

def blogcatalog():
    name_d = 'blogcatalog'
    dataset = 'data/'+name_d+'.mat'
    G = load_matfile(dataset)
    return G, name_d

def pos():
    name_d = 'POS'
    dataset = 'data/'+name_d+'.mat'
    G = load_matfile(dataset)
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

# G = lastfm()
# G = blogcatalog()
# G = flickr()
G, name_d = pubmed()


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
# embedder.wv.save_word2vec_format('pubmed.embeddings')
