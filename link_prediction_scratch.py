from scipy.io import loadmat
import random
from gensim.models import KeyedVectors, Word2Vec
from graph import *
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import numpy as np
from tqdm import tqdm
import csv

dataset = 'data/blogcatalog.mat'
embeddings_file = 'blogcatalog.embeddings'
nodes1 = []
nodes2 = []
all_nodes = {}
edges = {}
annot = loadmat(dataset)
net = annot['network']
cx = net.tocoo()
cont = 0
for i, j, v in tqdm(zip(cx.row, cx.col, cx.data)):
    nodes1.append(i)
    if i not in all_nodes:
        all_nodes[i] = 1
    else:
        all_nodes[i] += 1
    nodes2.append(j)
    if j not in all_nodes:
        all_nodes[j] = 1
    else:
        all_nodes[j] += 1
    edges[hash(str(i)+str(j))] = 0
    cont += 1


# test_size = round(len(all_nodes)*0.5)
test_size = round(len(nodes1)*0.5)

nodes_1_test = []
nodes_2_test = []

for i in tqdm(range(len(nodes1))):
    node = nodes1[i]
    node2 = nodes2[i]
    if all_nodes[node]>1 and all_nodes[node2]>1 and len(nodes_1_test)<=test_size:
        nodes_1_test.append(node)
        nodes_2_test.append(node2)
        nodes1[i] = None
        nodes2[i] = None


nodes_1_train = list(filter(None, nodes1))
nodes_2_train = list(filter(None, nodes2))

y_test = [1 for i in nodes_1_test]
y_train = [1 for i in nodes_1_train]
for i in tqdm(range(len(nodes_1_train))):
    negative_edge = [0, 0]
    while hash(str(negative_edge[0])+str(negative_edge[1])) in edges or negative_edge==[0, 0]:
        negative_edge = [random.randint(0, len(all_nodes)), random.randint(0, len(all_nodes))]
    nodes_1_train.append(negative_edge[0])
    nodes_2_train.append(negative_edge[1])
    y_train.append(0)

for i in tqdm(range(len(nodes_1_test))):
    negative_edge = [0, 0]
    while hash(str(negative_edge[0])+str(negative_edge[1])) in edges or negative_edge==[0, 0]:
        negative_edge = [random.randint(0, len(all_nodes)), random.randint(0, len(all_nodes))]
    nodes_1_test.append(negative_edge[0])
    nodes_2_test.append(negative_edge[1])
    y_test.append(0)



print("To embedding")

model = KeyedVectors.load_word2vec_format(embeddings_file, binary=False)


x_train = []
x_test = []
for i in range(len(nodes_1_train)):
    try:
        x_t = ([1/(1+ np.exp(-np.dot(model[str(nodes_1_train[i])], model[str(nodes_2_train[i])])))])
        x_train.append(x_t)
    except:
        del y_train[i]

for i in range(len(nodes_1_test)):
    try:
        x_t = ([1/(1+ np.exp(-np.dot(model[str(nodes_1_test[i])], model[str(nodes_2_test[i])])))])
        x_test.append(x_t)
    except:
        del y_test[i]

# x_train = [([1/(1+ np.exp(-np.dot(model[str(nodes_1_train[i])], model[str(nodes_2_train[i])])))]) for i in range(len(nodes_1_train))]
# x_test = [([1/(1+ np.exp(-np.dot(model[str(nodes_1_test[i])], model[str(nodes_2_test[i])])))]) for i in range(len(nodes_1_test))]

x_train = np.array(x_train)
x_test = np.array(x_test)

print("Finished")
lr = LogisticRegression(class_weight="balanced", verbose=False)


lr.fit(x_train, y_train)

predictions = lr.predict_proba(x_test)

print(roc_auc_score(y_test, predictions[:, 1]))