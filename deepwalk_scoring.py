import numpy
import sys
from collections import defaultdict
from gensim.models import KeyedVectors
from six import iteritems
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from scipy.io import loadmat
from sklearn.utils import shuffle as skshuffle
from sklearn.preprocessing import MultiLabelBinarizer
import networkx as nx

class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        assert X.shape[0] == len(top_k_list)
        probs = numpy.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            all_labels.append(labels)
        return all_labels

def to_graph(x):
    G = nx.Graph()
    cx = x.tocoo()
    for i,j,v in zip(cx.row, cx.col, cx.data):
        G.add_edge(*(i, j))
    return G

def main():
    # 0. Files
    name_d = 'flickr'
    embeddings_file = name_d+'.embeddings'
    matfile = 'data/'+name_d+'.mat'
    all = False

    # 1. Load Embeddings
    model = KeyedVectors.load_word2vec_format(embeddings_file, binary=False)

    # 2. Load labels
    mat = loadmat(matfile)
    A = mat['network']
    graph = to_graph(A)
    labels_matrix = mat['group']
    labels_count = labels_matrix.shape[1]
    mlb = MultiLabelBinarizer(range(labels_count))

    # Map nodes to their features (note:  assumes nodes are labeled as integers 1:N)
    features_matrix = numpy.asarray([model[str(node)] for node in range(len(graph))])

    # 2. Shuffle, to create train/test groups
    shuffles = []
    for x in range(5):
        shuffles.append(skshuffle(features_matrix, labels_matrix))

    # 3. to score each train/test group
    all_results = defaultdict(list)

    if all:
        training_percents = numpy.asarray(range(1, 10)) * .1
    else:
        training_percents = [0.1, 0.5, 0.9]
        # training_percents = [0.5]
    for train_percent in training_percents:
        for shuf in shuffles:

            X, y = shuf

            training_size = int(train_percent * X.shape[0])

            X_train = X[:training_size, :]
            y_train_ = y[:training_size]

            y_train = [[] for x in range(y_train_.shape[0])]

            cy = y_train_.tocoo()
            for i, j in zip(cy.row, cy.col):
                y_train[i].append(j)

            assert sum(len(l) for l in y_train) == y_train_.nnz

            X_test = X[training_size:, :]
            y_test_ = y[training_size:]

            y_test = [[] for _ in range(y_test_.shape[0])]

            cy = y_test_.tocoo()
            for i, j in zip(cy.row, cy.col):
                y_test[i].append(j)

            clf = TopKRanker(LogisticRegression())
            clf.fit(X_train, y_train_)

            # find out how many labels should be predicted
            top_k_list = [len(l) for l in y_test]
            preds = clf.predict(X_test, top_k_list)

            # y = y_train+y_test
            # top_k_list = [len(l) for l in y]
            # pred_t = clf.predict(X, top_k_list)
            # print(accuracy_score(mlb.fit_transform(y), mlb.fit_transform(pred_t)))
            results = {}
            averages = ["micro", "macro"]
            for average in averages:
                results[average] = f1_score(mlb.fit_transform(y_test), mlb.fit_transform(preds), average=average)

            all_results[train_percent].append(results)

    print('Results, using embeddings of dimensionality', X.shape[1])
    print('-------------------')
    for train_percent in sorted(all_results.keys()):
        print('Train percent:', train_percent)
        for index, result in enumerate(all_results[train_percent]):
            print('Shuffle #%d:   ' % (index + 1), result)
        avg_score = defaultdict(float)
        for score_dict in all_results[train_percent]:
            for metric, score in iteritems(score_dict):
                avg_score[metric] += score
        for metric in avg_score:
            avg_score[metric] /= len(all_results[train_percent])
        print('Average score:', dict(avg_score))
        print('-------------------')


if __name__ == "__main__":
    sys.exit(main())
