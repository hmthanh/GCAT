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
import pickle

def batch_gat_loss(gat_loss_func, train_indices, entity_embed, relation_embed):
    len_pos_triples = int(
        train_indices.shape[0] / (int(args.valid_invalid_ratio_gat) + 1))

    pos_triples = train_indices[:len_pos_triples]
    neg_triples = train_indices[len_pos_triples:]

    pos_triples = pos_triples.repeat(int(args.valid_invalid_ratio_gat), 1)

    source_embeds = entity_embed[pos_triples[:, 0]]
    relation_embeds = relation_embed[pos_triples[:, 1]]
    tail_embeds = entity_embed[pos_triples[:, 2]]

    x = source_embeds + relation_embeds - tail_embeds
    pos_norm = torch.norm(x, p=1, dim=1)

    source_embeds = entity_embed[neg_triples[:, 0]]
    relation_embeds = relation_embed[neg_triples[:, 1]]
    tail_embeds = entity_embed[neg_triples[:, 2]]

    x = source_embeds + relation_embeds - tail_embeds
    neg_norm = torch.norm(x, p=1, dim=1)

    y = -torch.ones(int(args.valid_invalid_ratio_gat) * len_pos_triples).cuda()

    loss = gat_loss_func(pos_norm, neg_norm, y)
    return loss