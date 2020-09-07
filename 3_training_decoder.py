import torch

from models import SpKBGATModified, SpKBGATConvOnly
from torch.autograd import Variable
import numpy as np
from utils import save_model, load_object, load_model, save_object
import random
import time
from create_config import Config

args = Config()
args.load_config()
device = "cuda" if args.cuda else "cpu"

print("Loading corpus")
Corpus_ = load_object(output=args.data_folder, name="corpus")
entity_embeddings = load_object(output=args.data_folder, name="entity_embeddings")
relation_embeddings = load_object(output=args.data_folder, name="relation_embeddings")
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
folder = "{output}/{dataset}".format(output=args.output_folder, dataset=args.dataset)
model_name = "{folder}/{dataset}_{device}_{name}_{epoch}.pt".format(folder=folder, dataset=args.dataset, device=device, name="gat", epoch=args.epochs_gat - 1)
model_gat.load_state_dict(torch.load(model_name), strict=False)

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
        num_iters_per_epoch = (len(Corpus_.train_indices) // args.batch_size_conv) + 1

    #for iters in range(num_iters_per_epoch):
    for iters in range(1):
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
    if epoch >= args.epochs_conv - 1:
        save_model(model_conv, name="conv", epoch=epoch)

save_object(epoch_losses, output=args.output_folder, name="loss_conv")

print("3. Train Decoder Successfully !")
