from math import ceil
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset

from datasets import node_classification
from layers import MeanAggregator, LSTMAggregator, MaxPoolAggregator, MeanPoolAggregator
import models
import utils
from sklearn.metrics import f1_score
import numpy as np

import hdf5storage as h5
import torchvision.transforms
import scipy.io

def main():

    # file = 'blogcatalog.mat'
    # data = scipy.io.loadmat(file)
    # A = data['network']
    # dataset = TensorDataset(torch.from_numpy(A.toarray()))


    config = utils.parse_args()

    if config['cuda'] and torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    dataset_args = (config['task'], config['dataset'], config['dataset_path'],
                    'train', config['num_layers'], config['self_loop'],
                    config['normalize_adj'], config['transductive'])
    dataset = utils.get_dataset(dataset_args)

    loader = DataLoader(dataset=dataset, batch_size=config['batch_size'],
                        shuffle=True, collate_fn=dataset.collate_wrapper)

    input_dim, output_dim = dataset.get_dims()
    print(input_dim,output_dim)

    agg_class = utils.get_agg_class(config['agg_class'])
    model = models.GraphSAGE(input_dim, config['hidden_dims'], output_dim, 
                             agg_class, config['dropout'],
                             config['num_samples'], device)
    model.to(device)

    # if not config['load']:
    #     criterion = utils.get_criterion(config['task'])
    #     optimizer = optim.Adam(model.parameters(), lr=config['lr'],
    #                         weight_decay=config['weight_decay'])
    #     epochs = config['epochs']
    #     print_every = config['print_every']
    #     num_batches = int(ceil(len(dataset) / config['batch_size']))
    #     model.train()
    #     print('--------------------------------')
    #     print('Training.')
    #     for epoch in range(epochs):
    #         print('Epoch {} / {}'.format(epoch+1, epochs))
    #         running_loss = 0.0
    #         num_correct, num_examples = 0, 0
    #         for (idx, batch) in enumerate(loader):
    #             features, node_layers, mappings, rows, labels = batch
    #             features, labels = features.to(device), labels.to(device)
    #             optimizer.zero_grad()
    #             out = model(features, node_layers, mappings, rows)
    #             loss = criterion(out, labels)
    #             loss.backward()
    #             optimizer.step()
    #             with torch.no_grad():
    #                 running_loss += loss.item()
    #                 predictions = torch.max(out, dim=1)[1]
    #                 num_correct += torch.sum(predictions == labels).item()
    #                 num_examples += len(labels)
    #             if (idx + 1) % print_every == 0:
    #                 running_loss /= print_every
    #                 accuracy = num_correct / num_examples
    #                 print('    Batch {} / {}: loss {}, accuracy {}'.format(
    #                     idx+1, num_batches, running_loss, accuracy))
    #                 running_loss = 0.0
    #                 num_correct, num_examples = 0, 0
    #                 # y_true = np.array(labels.cpu())
    #                 # y_pred = np.array(predictions.cpu())
    #                 # print(f1_score(y_true,y_pred,average='macro'))
    #     print('Finished training.')
    #     print('--------------------------------')

    #     if config['save']:
    #         print('--------------------------------')
    #         directory = os.path.join(os.path.dirname(os.getcwd()),
    #                                 'trained_models')
    #         if not os.path.exists(directory):
    #             os.makedirs(directory)
    #         fname = utils.get_fname(config)
    #         path = os.path.join(directory, fname)
    #         print('Saving model at {}'.format(path))
    #         torch.save(model.state_dict(), path)
    #         print('Finished saving model.')
    #         print('--------------------------------')

    # if config['load']:
    #     directory = os.path.join(os.path.dirname(os.getcwd()),
    #                              'trained_models')
    #     fname = utils.get_fname(config)
    #     path = os.path.join(directory, fname)
    #     model.load_state_dict(torch.load(path))
    # dataset_args = (config['task'], config['dataset'], config['dataset_path'],
    #                 'test', config['num_layers'], config['self_loop'],
    #                 config['normalize_adj'], config['transductive'])
    # dataset = utils.get_dataset(dataset_args)
    # loader = DataLoader(dataset=dataset, batch_size=config['batch_size'],
    #                     shuffle=False, collate_fn=dataset.collate_wrapper)
    # criterion = utils.get_criterion(config['task'])
    # print_every = config['print_every']
    # num_batches = int(ceil(len(dataset) / config['batch_size']))
    # model.eval()
    # print('--------------------------------')
    # print('Testing.')
    # running_loss, total_loss = 0.0, 0.0
    # num_correct, num_examples = 0, 0
    # total_correct, total_examples = 0, 0
    # f1_macro = np.array([])
    # f1_micro = np.array([])
    # for (idx, batch) in enumerate(loader):
    #     features, node_layers, mappings, rows, labels = batch
    #     features, labels = features.to(device), labels.to(device)
    #     out = model(features, node_layers, mappings, rows)
    #     loss = criterion(out, labels)
    #     running_loss += loss.item()
    #     total_loss += loss.item()
    #     predictions = torch.max(out, dim=1)[1]
    #     num_correct += torch.sum(predictions == labels).item()
    #     total_correct += torch.sum(predictions == labels).item()
    #     num_examples += len(labels)
    #     total_examples += len(labels)
    #     if (idx + 1) % print_every == 0:
    #         running_loss /= print_every
    #         accuracy = num_correct / num_examples
    #         print('    Batch {} / {}: loss {}, accuracy {}'.format(
    #             idx+1, num_batches, running_loss, accuracy))
    #         running_loss = 0.0
    #         num_correct, num_examples = 0, 0
    #     y_true = np.array(labels.cpu())
    #     y_pred = np.array(predictions.cpu())
    #     f1_macro = np.append(f1_macro,f1_score(y_true,y_pred,average='macro'))
    #     f1_micro = np.append(f1_micro,f1_score(y_true,y_pred,average='micro'))
    # print('Macro F1 Score {}'.format(f1_macro.mean()))
    # print('Micro F1 Score {}'.format(f1_micro.mean()))
    # total_loss /= num_batches
    # total_accuracy = total_correct / total_examples
    # print('Loss {}, accuracy {}'.format(total_loss, total_accuracy))
    # print()
    # print('Finished testing.')
    # print('--------------------------------')

if __name__ == '__main__':
    main()
