from scipy.io import loadmat
import random
from gensim.models import KeyedVectors
from graph import *
import pandas as pd
import networkx as nx
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from tqdm import tqdm

def to_graph(x):
    G = nx.Graph()
    cx = x.tocoo()
    for i, j, v in zip(cx.row, cx.col, cx.data):
        G.add_edge(*(i, j))
    return G


dataset = 'data/blogcatalog.mat'
embeddings_file = 'blogcatalog2.embeddings'
graph = load_matfile(dataset)

nodes1 = []
nodes2 = []
annot = loadmat(dataset)
net = annot['network']
cx = net.tocoo()
for i,j,v in zip(cx.row, cx.col, cx.data):
    nodes1.append(str(i))
    nodes2.append(str(j))

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
G_data = nx.from_pandas_edgelist(fb_df_partial, "node_1", "node_2", create_using=nx.Graph())

model = KeyedVectors.load_word2vec_format(embeddings_file, binary=False)

x = [(model[str(i)] + model[str(j)]) for i, j in zip(data['node_1'], data['node_2'])]

xtrain, xtest, ytrain, ytest = train_test_split(np.array(x), data['link'],
                                                test_size=0.5,
                                                random_state=42)

lr = LogisticRegression(class_weight="balanced")

lr.fit(xtrain, ytrain)

predictions = lr.predict_proba(xtest)

print(roc_auc_score(ytest, predictions[:,1]))