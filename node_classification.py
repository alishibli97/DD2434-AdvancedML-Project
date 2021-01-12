from sklearn.linear_model import LogisticRegression
from tensorflow.python import keras

from utils import svm_classify, one_vs_rest_classify, load_data, load_data_mat
import numpy as np
from sklearn.model_selection import cross_val_score

label_file = 'dblp/labels.txt'
edge_file = 'dblp/adjedges.txt'
adj_list, labels_dict = load_data(label_file, edge_file)
model = keras.models.load_model('model_dblp', compile=False)
embed_generator = keras.models.load_model('embed_generator_dblp', compile=False)

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

clf = LogisticRegression()
scores = cross_val_score(clf, new_X, new_label.ravel(), cv=5, scoring='f1_macro')
print('f1_macro: ')
print(scores.mean())
scores = cross_val_score(clf, new_X, new_label.ravel(), cv=5, scoring='f1_micro')
print('f1_micro: ')
print(scores.mean())
