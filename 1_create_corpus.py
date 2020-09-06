import torch

from models import SpKBGATModified, SpKBGATConvOnly
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from copy import deepcopy

from preprocess import read_entity_from_id, read_relation_from_id, init_embeddings, build_data
from create_batch import Corpus
from utils import save_model
import random
import argparse
import os
import sys
import logging
import time

from create_config import Config

args = Config()
args.load_config()

train_data, validation_data, test_data, entity2id, relation2id, headTailSelector, unique_entities_train = build_data(
        args.data_folder, is_unweigted=False, directed=True)

#(train_triples, train_adjacency_mat) = train_data
(test_triples, test_adjacency_mat) = test_data
print("test_triples", np.array(test_triples).shape)
print("test_adjacency_mat", np.array(test_adjacency_mat).shape)
print("entity2id", len(entity2id))
print("relation2id", len(relation2id))
print("headTailSelector", len(headTailSelector))
print("unique_entities_train", np.array(unique_entities_train).shape, "\n")

def load_data_main(args):
    train_data, validation_data, test_data, entity2id, relation2id, headTailSelector, unique_entities_train = build_data(
        args.data_folder, is_unweigted=False, directed=True)

    if args.pretrained_emb:
        entity_embeddings, relation_embeddings = init_embeddings(os.path.join(args.data_folder, 'entity2vec.txt'),
                                                                 os.path.join(args.data_folder, 'relation2vec.txt'))
        print("Initialised relations and entities from TransE")

    else:
        entity_embeddings = np.random.randn(
            len(entity2id), args.embedding_size)
        relation_embeddings = np.random.randn(
            len(relation2id), args.embedding_size)
        print("Initialised relations and entities randomly")

    corpus = Corpus(args, train_data, validation_data, test_data, entity2id, relation2id, headTailSelector,
                    args.batch_size_gat, args.valid_invalid_ratio_gat, unique_entities_train, args.get_2hop)

    return corpus, torch.FloatTensor(entity_embeddings), torch.FloatTensor(relation_embeddings)

Corpus_, entity_embeddings, relation_embeddings = load_data_main(args)


if (args.save_gdrive):
    torch.save(Corpus_, args.drive_folder + "Corpus_torch.pt")
    torch.save(entity_embeddings, args.drive_folder + "entity_embeddings.pt")
    torch.save(relation_embeddings, args.drive_folder + "relation_embeddings.pt")
    node_neighbors_2hop = Corpus_.node_neighbors_2hop
else:
    torch.save(Corpus_, args.data_folder + "Corpus_torch.pt")
    torch.save(entity_embeddings, args.data_folder + "entity_embeddings.pt")
    torch.save(relation_embeddings, args.data_folder + "relation_embeddings.pt")
    node_neighbors_2hop = Corpus_.node_neighbors_2hop

print("1. Created Corpus Successfully !")
