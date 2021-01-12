from scipy.io import loadmat
import random
from gensim.models import KeyedVectors, Word2Vec
from graph import *
import pandas as pd
import networkx as nx
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from tqdm import tqdm
import csv
from scipy.sparse import csr_matrix
import pickle



def to_graph(x):
    G = nx.Graph()
    cx = x.tocoo()
    for i, j, v in zip(cx.row, cx.col, cx.data):
        G.add_edge(*(i, j))
    return G

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
    csv_network = 'data/lastfm_asia_edges.csv'
    graph = get_graph_csv(csv_network)
    embeddings_file = 'lastfm' + '.embeddings'
    rows = []
    cols = []
    cont = 0
    with open(csv_network, newline='', ) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            # if cont == 2000: break
            rows.append(int(row[0]))
            cols.append(int(row[1]))
            cont += 1
    data = [1.0 for i in rows]
    cx = csr_matrix((data, (rows, cols)))
    cx = cx.tocoo()
    nodes1 = []
    nodes2 = []
    for i, j, v in zip(cx.row, cx.col, cx.data):
        nodes1.append(str(i))
        nodes2.append(str(j))
    return nodes1, nodes2, graph, embeddings_file

def blogcatalog():
    dataset = 'data/blogcatalog.mat'
    embeddings_file = 'blogcatalog.embeddings'
    graph = load_matfile(dataset)
    nodes1 = []
    nodes2 = []
    annot = loadmat(dataset)
    net = annot['network']
    cx = net.tocoo()
    cont = 0
    for i,j,v in zip(cx.row, cx.col, cx.data):
        if cont==2000: break
        nodes1.append(str(i))
        nodes2.append(str(j))
        cont+=1
    return nodes1, nodes2, graph, embeddings_file

def pos():
    dataset = 'data/POS.mat'
    embeddings_file = 'POS.embeddings'
    graph = load_matfile(dataset)
    nodes1 = []
    nodes2 = []
    annot = loadmat(dataset)
    net = annot['network']
    cx = net.tocoo()
    cont = 0
    for i,j,v in zip(cx.row, cx.col, cx.data):
        # if cont==2000: break
        nodes1.append(str(i))
        nodes2.append(str(j))
        cont+=1
    return nodes1, nodes2, graph, embeddings_file

def sapiens():
    dataset = 'data/Homo_sapiens.mat'
    embeddings_file = 'Homo_sapiens.embeddings'
    graph = load_matfile(dataset)
    nodes1 = []
    nodes2 = []
    annot = loadmat(dataset)
    net = annot['network']
    cx = net.tocoo()
    cont = 0
    for i,j,v in zip(cx.row, cx.col, cx.data):
        # if cont==2000: break
        nodes1.append(str(i))
        nodes2.append(str(j))
        cont+=1
    return nodes1, nodes2, graph, embeddings_file



nodes1, nodes2, graph, embeddings_file = blogcatalog()


fb_df = pd.DataFrame({'node_1': nodes1, 'node_2': nodes2})

G = nx.from_pandas_edgelist(fb_df, "node_1", "node_2", create_using=nx.Graph())
# combine all nodes in a list
node_list = nodes1 + nodes2

# remove duplicate items from the list
node_list = list(dict.fromkeys(node_list))

# build adjacency matrix
adj_G = nx.to_numpy_matrix(G, nodelist=node_list)

# get unconnected node-pairs
all_unconnected_pairs = []

# traverse adjacency matrix
offset = 1
for i in tqdm(range(1, adj_G.shape[0]-1)):
    for j in range(offset, adj_G.shape[1]-1):
        if i != j:
            # if nx.shortest_path_length(G, str(i), str(j)) <= 2:
            if j in graph[i]:
                if adj_G[i, j] == 0:
                    all_unconnected_pairs.append([node_list[i], node_list[j]])

    offset = offset + 1

node_1_unlinked = [i[0] for i in all_unconnected_pairs]
node_2_unlinked = [i[1] for i in all_unconnected_pairs]

data = pd.DataFrame({'node_1': node_1_unlinked,
                     'node_2': node_2_unlinked})

# add target variable 'link'
data['link'] = 0

initial_node_count = len(G.nodes)

fb_df_temp = fb_df.copy()

# empty list to store removable links
omissible_links_index = []

for i in tqdm(fb_df.index.values):

    # remove a node pair and build a new graph
    G_temp = nx.from_pandas_edgelist(fb_df_temp.drop(index=i), "node_1", "node_2", create_using=nx.Graph())

    # check there is no spliting of graph and number of nodes is same
    if (nx.number_connected_components(G_temp) == 1) and (len(G_temp.nodes) == initial_node_count):
        omissible_links_index.append(i)
        fb_df_temp = fb_df_temp.drop(index=i)

# create dataframe of removable edges
fb_df_ghost = fb_df.loc[omissible_links_index]

# add the target variable 'link'
fb_df_ghost['link'] = 1

data = data.append(fb_df_ghost[['node_1', 'node_2', 'link']], ignore_index=True)

# drop removable edges
fb_df_partial = fb_df.drop(index=fb_df_ghost.index.values)

# build graph

# G_data = nx.from_pandas_edgelist(fb_df_partial, "node_1", "node_2", create_using=nx.Graph())
# G = Graph()
# for i in range(len(fb_df_partial.node_1)):
#     n1 = list(fb_df_partial.node_1)[i]
#     n2 = list(fb_df_partial.node_2)[i]
#     G[n1].append(n2)
# G.make_undirected()
# G.make_consistent()
# t = 40  # walk length
# r = 80  # number of walks
# ns = 5  # negative sample size
# w = 10  # window size
# worker_threads = 32
# embedding_dim = 128
# seed = 42
#
# walks = build_deepwalk_corpus(G, num_paths=r, path_length=t, alpha=0, rand=random.Random(seed))
#
# model = Word2Vec(walks, window=w, sg=1, hs=1, negative=ns,
#                     seed=42, size=embedding_dim, workers=worker_threads, min_count=0)


model = KeyedVectors.load_word2vec_format(embeddings_file, binary=False)

x = [(model[str(i)] + model[str(j)]) for i, j in zip(data['node_1'], data['node_2'])]

# with open('x.pickle', 'wb') as fp:
#     pickle.dump(x, fp)
#
# with open('data.pickle', 'wb') as fp:
#     pickle.dump(data, fp)
    

# with open('x.pickle', 'rb') as handle:
#     x = pickle.load(handle)
#
# with open('data.pickle', 'rb') as handle:
#     data = pickle.load(handle)


xtrain, xtest, ytrain, ytest = train_test_split(np.array(x), data['link'],
                                                test_size=0.05,
                                                random_state=42, shuffle=True)

lr = LogisticRegression(class_weight="balanced")

lr.fit(xtrain, ytrain)

predictions = lr.predict_proba(xtest)

print(roc_auc_score(ytest, predictions[:,1]))