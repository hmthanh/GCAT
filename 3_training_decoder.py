import torch

from models import SpKBGATModified, SpKBGATConvOnly
from torch.autograd import Variable
import numpy as np
from utils import save_model

import random
import time
from create_config import Config

args = Config()
args.load_config()

print("Loading corpus")
device = "gpu" if args.cuda else "cpu"
Corpus_ = torch.load("{output}corpus_{device}.pt".format(output=args.data_folder, device=device))
entity_embeddings = torch.load("{output}entity_embeddings_{device}.pt".format(output=args.data_folder, device=device))
relation_embeddings = torch.load("{output}relation_embeddings_{device}.pt".format(output=args.data_folder, device=device))
node_neighbors_2hop = Corpus_.node_neighbors_2hop

print("Defining model")
model_gat = SpKBGATModified(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                            args.drop_GAT, args.alpha, args.nheads_GAT)
model_conv = SpKBGATConvOnly(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                             args.drop_GAT, args.drop_conv, args.alpha, args.alpha_conv,
                             args.nheads_GAT, args.out_channels)

if args.cuda:
    model_conv.cuda()
    model_gat.cuda()

print("Loading GAT encoder")
gat_model_path = '{output}{dataset}/trained_{epoch}.pt'.format(output=args.output_folder, dataset=args.dataset, epoch=args.epochs_gat - 1)
model_gat.load_state_dict(torch.load(gat_model_path), strict=False)

print("Only Conv model trained")
model_conv.final_entity_embeddings = model_gat.final_entity_embeddings
model_conv.final_relation_embeddings = model_gat.final_relation_embeddings

Corpus_.batch_size = args.batch_size_conv
Corpus_.invalid_valid_ratio = int(args.valid_invalid_ratio_conv)

optimizer = torch.optim.Adam(
model_conv.parameters(), lr=args.lr, weight_decay=args.weight_decay_conv)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5, last_epoch=-1)

margin_loss = torch.nn.SoftMarginLoss()

epoch_losses = []   # losses of all epochs
print("Number of epochs {}".format(args.epochs_conv))

for epoch in range(args.epochs_conv):
    print("\nepoch-> ", epoch)
    random.shuffle(Corpus_.train_triples)
    Corpus_.train_indices = np.array(
        list(Corpus_.train_triples)).astype(np.int32)

    model_conv.train()  # getting in training mode
    start_time = time.time()
    epoch_loss = []

    if len(Corpus_.train_indices) % args.batch_size_conv == 0:
        num_iters_per_epoch = len(
            Corpus_.train_indices) // args.batch_size_conv
    else:
        num_iters_per_epoch = (
                                      len(Corpus_.train_indices) // args.batch_size_conv) + 1

    for iters in range(num_iters_per_epoch):
        start_time_iter = time.time()
        train_indices, train_values = Corpus_.get_iteration_batch(iters)

        if args.cuda:
            train_indices = Variable(
                torch.LongTensor(train_indices)).cuda()
            train_values = Variable(torch.FloatTensor(train_values)).cuda()

        else:
            train_indices = Variable(torch.LongTensor(train_indices))
            train_values = Variable(torch.FloatTensor(train_values))

        preds = model_conv(
            Corpus_, Corpus_.train_adj_matrix, train_indices)

        optimizer.zero_grad()

        loss = margin_loss(preds.view(-1), train_values.view(-1))

        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.data.item())

        end_time_iter = time.time()

        print("Iteration-> {0}  , Iteration_time-> {1:.4f} , Iteration_loss {2:.4f}".format(
            iters, end_time_iter - start_time_iter, loss.data.item()))

    scheduler.step()
    print("Epoch {} , average loss {} , epoch_time {}".format(
        epoch, sum(epoch_loss) / len(epoch_loss), time.time() - start_time))
    epoch_losses.append(sum(epoch_loss) / len(epoch_loss))
    if (epoch > args.epochs_conv - 3):
        save_model(model_conv, args.data_folder, epoch, args.output_folder)

torch.save(epoch_losses, args.output_folder + "epoch_losses_conv.pt")

print("3. Train Decoder Successfully !")
