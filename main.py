from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from utils import svm_classify, one_vs_rest_classify, batchgen_train, LINE_loss, load_data, load_data_mat, load_data_csv
import numpy as np
from model import create_model
from tensorflow.keras.callbacks import LearningRateScheduler
import random


class LearningRateDecay:
    def __init__(self, maxEpochs=100, initAlpha=0.01, power=1.0):
        # store the maximum number of epochs, base learning rate,
        # and power of the polynomial
        self.maxEpochs = maxEpochs
        self.initAlpha = initAlpha
        self.power = power

    def __call__(self, epoch):
        # compute the new learning rate based on polynomial decay
        decay = (1 - (epoch / float(self.maxEpochs))) ** self.power
        alpha = self.initAlpha * decay
        # return the new learning rate
        return float(alpha)


if __name__ == "__main__":

    # label_file = 'dblp/labels.txt'
    # edge_file = 'dblp/adjedges.txt'
    label_file = 'datasets/Pubmed-Diabetes/pubmed_labels.csv'
    edge_file = 'datasets/Pubmed-Diabetes/pubmed_edges.csv'
    # data_file = 'datasets/blogcatalog.mat'
    epoch_num = 100000
    init_alpha = 0.025
    factors = 128
    batch_size = 1000
    negative_sampling = "UNIFORM"  # UNIFORM or NON-UNIFORM
    negativeRatio = 5
    split_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # svm_C = 0.1

    np.random.seed(2021)
    random.seed(2021)

    # adj_list, labels_dict = load_data(label_file, edge_file)
    adj_list, labels_dict = load_data_csv(label_file, edge_file)
    print(adj_list)
    print()
    print(labels_dict)
    exit()
    # adj_list, labels_dict = load_data_mat(data_file)
    # epoch_train_size = (((int(len(adj_list) / batch_size)) * (1 + negativeRatio) * batch_size) + (1 + negativeRatio) * (len(adj_list) % batch_size))
    epoch_train_size = 1
    numNodes = np.max(adj_list.ravel()) + 1
    data_gen = batchgen_train(adj_list, numNodes, batch_size, negativeRatio, negative_sampling)

    model, embed_generator = create_model(numNodes, factors)
    model.summary()

    model.compile(optimizer='sgd', loss={'left_right_dot': LINE_loss})

    schedule = LearningRateDecay(maxEpochs=epoch_num, initAlpha=init_alpha, power=1)
    callbacks = [LearningRateScheduler(schedule)]
    model.fit_generator(data_gen, samples_per_epoch=epoch_train_size, nb_epoch=epoch_num, callbacks=callbacks, verbose=0)
    model.save('model_pubmed')
    embed_generator.save('embed_generator_pubmed')

    # model = keras.models.load_model('model', compile=False)
    # embed_generator = keras.models.load_model('embed_generator', compile=False)

    new_X = []
    new_label = []

    keys = list(labels_dict.keys())
    np.random.shuffle(keys)

    for k in keys:
        v = labels_dict[k]
        x = embed_generator.predict_on_batch([np.asarray([k]), np.asarray([k])])
        new_X.append(x[0][0] + x[1][0])  # dimension: same as factors
        new_label.append(labels_dict[k])

    new_X = np.asarray(new_X, dtype=np.float32)
    new_label = np.asarray(new_label, dtype=np.int32)

    # train_macro_f1, train_micro_f1, test_macro_f1, test_micro_f1 = svm_classify(new_X, new_label, split_ratios, svm_C)
    # train_macro_f1, train_micro_f1, test_macro_f1, test_micro_f1 = one_vs_rest_classify(new_X, new_label, split_ratios)
    #
    # print('Results, using embeddings of dimensionality', factors)
    # print('-------------------')
    # for i in range(len(split_ratios)):
    #     print('Train percent:', split_ratios[i])
    #     # print('Training set average score: "macro": %.4f, "micro": %.4f' % (train_macro_f1[i], train_micro_f1[i]))
    #     print('Test set average score: "macro": %.4f, "micro": %.4f' % (test_macro_f1[i], test_micro_f1[i]))
    #     print('-------------------')

clf = LogisticRegression()
scores = cross_val_score(clf, new_X, new_label.ravel(), cv=5, scoring='f1_macro')
print('f1_macro: ')
print(scores.mean())
scores = cross_val_score(clf, new_X, new_label.ravel(), cv=5, scoring='f1_micro')
print('f1_micro: ')
print(scores.mean())