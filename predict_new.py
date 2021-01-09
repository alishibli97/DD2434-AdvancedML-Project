#!/usr/bin/env python
# encoding: utf-8
# File Name: predict.py
# Author: Jiezhong Qiu
# Create Time: 2017/07/17 21:57
# TODO:

import os
import pickle as pkl
import numpy as np
import scipy.io
import argparse
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.exceptions import UndefinedMetricWarning
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

logger = logging.getLogger(__name__)

def construct_indicator(y_score, y):
    # rank the labels by the scores directly
    num_label = np.sum(y, axis=1, dtype=np.int)
    y_sort = np.fliplr(np.argsort(y_score, axis=1))
    y_pred = np.zeros_like(y, dtype=np.int)
    for i in range(y.shape[0]):
        for j in range(num_label[i]):
            y_pred[i, y_sort[i, j]] = 1
    return y_pred

def load_w2v_feature(file):
    with open(file, "rb") as f:
        nu = 0
        for line in f:
            content = line.strip().split()
            nu += 1
            if nu == 1:
                n, d = int(content[0]), int(content[1])
                feature = [[] for i in range(n)]
                continue
            index = int(content[0])
            for x in content[1:]:
                feature[index].append(float(x))
    for item in feature:
        assert len(item) == d
    return np.array(feature, dtype=np.float32)


def load_label(file, variable_name="group"):
    data = scipy.io.loadmat(file)
    logger.info("loading mat file %s", file)
    label = data[variable_name].todense().astype(np.int)
    label = np.array(label)
#     print(label.shape, type(label), label.min(), label.max())
    return label

def predict_cv(X, y, train_ratio=0.2, n_splits=10, random_state=0, C=1.):
    micro, macro = [], []
    shuffle = ShuffleSplit(n_splits=n_splits, test_size=1-train_ratio,
            random_state=random_state)
    for train_index, test_index in shuffle.split(X):
        print(train_index.shape, test_index.shape)
        assert len(set(train_index) & set(test_index)) == 0
        assert len(train_index) + len(test_index) == X.shape[0]
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = OneVsRestClassifier(
                LogisticRegression(
                    C=C,
                    solver="liblinear",
                    multi_class="ovr"),
                n_jobs=-1)
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)
        y_pred = construct_indicator(y_score, y_test)
        mi = f1_score(y_test, y_pred, average="micro")
        ma = f1_score(y_test, y_pred, average="macro")
        logger.info("micro f1 %f macro f1 %f", mi, ma)
        micro.append(mi)
        macro.append(ma)
    logger.info("%d fold validation, training ratio %f", len(micro), train_ratio)
    logger.info("Average micro %.2f, Average macro %.2f",
            np.mean(micro) * 100,
            np.mean(macro) * 100)


if __name__ == "__main__":

        file = 'PubMed'

        embedding = 'netmf_pubmed.npy'
        
        matfile_variable_name = 'group'
        seed = 0
        start_train_ratio = 10
        stop_train_ratio = 90
        num_train_ratio = 9
        C = 1.0
        num_split = 5

         # loading labels
        logging.basicConfig(
            #filename="%s.log" % args.embedding, filemode="w", # uncomment this to log to file
            level=logging.INFO,
            format='%(asctime)s %(message)s') # include timestamp
        logger.info("Loading label from %s...", file)

        data = open(file+'.edges').readlines()
        data = np.array([[int(dd) for dd in d.strip().split(',')] for d in data])

        row = data[:,0]
        col = data[:,1]

        vertices = set(np.concatenate((row,col),axis=0))

        map_vertex = {vertex:i for i,vertex in enumerate(vertices)}

        label = open(file+'.node_labels').readlines()
        label = np.array([[int(dd) for dd in d.strip().split(',')] for d in label])

        labels = np.zeros((len(map_vertex),max(label[:,1])),dtype=int)

        for i,r in enumerate(label):
            labels[map_vertex[r[0]]][r[1]-1] = 1
        logger.info("Label loaded!")

        # loading embedding
        logger.info("Loading network embedding from %s...", embedding)
        embedding = np.load(embedding)
        logger.info("Network embedding loaded!")

        train_ratios = np.linspace(start_train_ratio, stop_train_ratio, num_train_ratio)
        for tr in train_ratios:
            predict_cv(embedding, labels, train_ratio=tr/100., n_splits=num_split, C=C, random_state=seed)

