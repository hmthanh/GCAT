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

# import torch_xla
# import torch_xla.core.xla_model as xm
# TPU = xm.xla_device()
#@title Cấu hình cài đặt quá trình huấn luyện { run: "auto" }
#@markdown Thư mục hiện tại
#@title Cấu hình cài đặt quá trình huấn luyện { run: "auto" }
#@markdown Thư mục hiện tại
root_folder = "D:/thesis/CGAT/" #@param {type: "string"}

#@markdown Siêu tham số

dataset = "WN18" #@param ["WN18RR", "WN18", "FB15k", "FB15k-237"] {allow-input: false}
Use_Google_Drive_Save_Data = False #@param {type: "boolean"}
drive_folder = "drive/My Drive/" #@param {type: "string"}

CUDA = False #@param {type: "boolean"}
epochs_gat = 3600 #@param {type: "number"}
epochs_conv = 200 #@param {type: "number"}
weight_decay_gat = float(5e-6) #@param {type:"raw"}
weight_decay_conv = float(1e-5) #@param {type:"raw"}
pretrained_emb = True #@param {type: "boolean"}
embedding_size = 50 #@param {type: "slider", min: 20, max: 200}
lr = float(1e-3) #@param {type:"raw"}
get_2hop = True #@param {type: "boolean"}
use_2hop = True #@param {type: "boolean"}
partial_2hop = False #@param {type: "boolean"}
output_folder = "output/"

#@markdown --- Tham số cho mô hình GAT
batch_size_gat = 86835 #@param {type: "number"}
valid_invalid_ratio_gat = 2 #@param {type: "slider", min: 1, max: 10}
drop_GAT = 0.3  #@param {type:"raw"}
alpha = 0.2  #@param {type:"raw"}
entity_out_dim = [100, 200]  #@param {type:"raw"}
nheads_GAT = [2, 2]  #@param {type:"raw"}
margin = 5 #@param {type: "slider", min: 1, max: 10}

#@markdown --- Tham số cho mô hình ConvKB
batch_size_conv = 128  #@param {type: "slider", min: 16, max: 512, step:16}
alpha_conv = 0.2  #@param {type:"raw"}
valid_invalid_ratio_conv = 40
out_channels = 500 #@param {type: "slider", min: 100, max: 1000, step:100}
drop_conv = 0.0  #@param {type:"raw"}

class Args:
    dataset = dataset
    data = "{root_folder}data/{dataset}/".format(root_folder=root_folder, dataset = dataset)
    drive = root_folder + drive_folder
    output_folder = "{root_folder}output/{dataset}/".format(root_folder=root_folder, dataset=dataset)
    if (Use_Google_Drive_Save_Data):
        output_folder = drive

    epochs_gat = epochs_gat
    epochs_conv = epochs_conv
    weight_decay_gat = weight_decay_gat
    weight_decay_conv = weight_decay_conv
    pretrained_emb = pretrained_emb
    embedding_size = embedding_size
    lr = lr
    get_2hop = get_2hop
    use_2hop = use_2hop
    partial_2hop = partial_2hop
    output_folder = output_folder
    

    # Tham số cho mô hình GAT
    batch_size_gat = batch_size_gat
    valid_invalid_ratio_gat = valid_invalid_ratio_gat
    drop_GAT = drop_GAT
    alpha = alpha
    entity_out_dim = entity_out_dim
    nheads_GAT = nheads_GAT
    margin = margin

    # Tham số cho mô hình ConvKB
    batch_size_conv = batch_size_conv
    alpha_conv = alpha_conv
    valid_invalid_ratio_conv = valid_invalid_ratio_conv
    out_channels = out_channels
    drop_conv = drop_conv

args = Args()

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

Corpus_ = torch.load(args.data_folder + "Corpus_torch.pt")
entity_embeddings = torch.load(args.data_folder + "entity_embeddings.pt")
relation_embeddings = torch.load(args.data_folder + "relation_embeddings.pt")
node_neighbors_2hop = Corpus_.node_neighbors_2hop

print("Defining model")
model_gat = SpKBGATModified(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                            args.drop_GAT, args.alpha, args.nheads_GAT)
print("Only Conv model trained")
model_conv = SpKBGATConvOnly(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                              args.drop_GAT, args.drop_conv, args.alpha, args.alpha_conv,
                              args.nheads_GAT, args.out_channels)

if CUDA:
        model_conv.cuda()
        model_gat.cuda()
		
optimizer = torch.optim.Adam(
    model_gat.parameters(), lr=args.lr, weight_decay=args.weight_decay_gat)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=500, gamma=0.5, last_epoch=-1)

gat_loss_func = nn.MarginRankingLoss(margin=args.margin)

current_batch_2hop_indices = torch.tensor([])
if(args.use_2hop):
    current_batch_2hop_indices = Corpus_.get_batch_nhop_neighbors_all(args, Corpus_.unique_entities_train, node_neighbors_2hop)

if CUDA:
    current_batch_2hop_indices = Variable(torch.LongTensor(current_batch_2hop_indices)).cuda()
else:
    current_batch_2hop_indices = Variable(torch.LongTensor(current_batch_2hop_indices))
    

epoch_losses = []   # losses of all epochs
print("Number of epochs {}".format(args.epochs_gat))

for epoch in range(args.epochs_gat):
    random.shuffle(Corpus_.train_triples)
    Corpus_.train_indices = np.array(
        list(Corpus_.train_triples)).astype(np.int32)

    model_gat.train()  # getting in training mode
    start_time = time.time()
    epoch_loss = []

    if len(Corpus_.train_indices) % args.batch_size_gat == 0:
        num_iters_per_epoch = len(
            Corpus_.train_indices) // args.batch_size_gat
    else:
        num_iters_per_epoch = (
            len(Corpus_.train_indices) // args.batch_size_gat) + 1

    for iters in range(num_iters_per_epoch):
        start_time_iter = time.time()
        train_indices, train_values = Corpus_.get_iteration_batch(iters)

        if CUDA:
            train_indices = Variable(
                torch.LongTensor(train_indices)).cuda()
            train_values = Variable(torch.FloatTensor(train_values)).cuda()
        else:
            train_indices = Variable(torch.LongTensor(train_indices))
            train_values = Variable(torch.FloatTensor(train_values))

        # forward pass
        entity_embed, relation_embed = model_gat(
            Corpus_, Corpus_.train_adj_matrix, train_indices, current_batch_2hop_indices)

        optimizer.zero_grad()

        loss = batch_gat_loss(
            gat_loss_func, train_indices, entity_embed, relation_embed)

        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.data.item())

        end_time_iter = time.time()

    scheduler.step()
    epoch_losses.append(sum(epoch_loss) / len(epoch_loss))

    if (epoch > args.epochs_gat - 3):
        save_model(model_gat, args.data_folder, epoch, args.output_folder)
        torch.save(epoch_losses, args.output_folder + "epoch_losses.pt")
		
