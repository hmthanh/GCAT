import matplotlib.pyplot as plt
import numpy as np
import torch
# from torchviz import make_dot, make_dot_from_trace
import torch.nn as nn
from matplotlib.lines import Line2D

from config import Config

args = Config()
args.load_config()
device = "cuda" if args.cuda else "cpu"

def save_model(model, name, epoch):
    output = args.output_folder

    model_name = "{name}_{epoch}".format(name=name, epoch=epoch)
    save_object(model.state_dict(), output, model_name)
    print("Done saving model {name}".format(name=name))


def load_model(name, epoch):
    output = args.output_folder
    # folder = "{output}/{dataset}".format(output=args.output_folder, dataset=args.dataset)
    # model_name = "{folder}/{dataset}_{device}_{name}_{epoch}.pt".format(folder=folder, dataset=args.dataset, device=device, name="gat", epoch=args.epochs_gat - 1)
    model_name = "{name}_{epoch}".format(name=name, epoch=epoch)
    load_object(output=output, name=model_name)
    print("Done loading model {name}".format(name=name))


def save_object(obj, output, name):
    folder = "{output}/{dataset}".format(output=output, dataset=args.dataset)
    if args.save_gdrive:
        folder = args.drive_folder
    torch.save(obj, "{folder}/{dataset}_{device}_{name}.pt".format(folder=folder, dataset=args.dataset, name=name, device=device))


def load_object(output, name):
    folder = "{output}/{dataset}".format(output=output, dataset=args.dataset)
    if args.save_gdrive:
        folder = args.drive_folder
    
    file_path = "{folder}/{dataset}_{device}_{name}.pt".format(folder=folder, dataset=args.dataset, name=name, device=device)
    # print(file_path)
    return torch.load(file_path)


def save_txt(output, file, lines):
    folder = "{output}/{dataset}".format(output=output, dataset=args.dataset)
    if args.save_gdrive:
        folder = args.drive_folder
    txt_name = "{folder}/{dataset}_{device}_{file}".format(folder=folder, dataset=args.dataset, file=file,
                                                           device=device)
    with open(txt_name, "w") as f:
        f.write(lines)


def load_txt(output, file="result"):
    folder = "{output}/{dataset}".format(output=output, dataset=args.dataset)
    if args.save_gdrive:
        folder = args.drive_folder
    txt_name = "{folder}/{dataset}_{device}_{file}".format(folder=folder, dataset=args.dataset, file=file,
                                                           device=device)
    with open(txt_name) as f:
        return f.read()


gat_loss_func = nn.MarginRankingLoss(margin=0.5)


def GAT_Loss(train_indices, valid_invalid_ratio):
    len_pos_triples = train_indices.shape[0] // (int(valid_invalid_ratio) + 1)

    pos_triples = train_indices[:len_pos_triples]
    neg_triples = train_indices[len_pos_triples:]

    pos_triples = pos_triples.repeat(int(valid_invalid_ratio), 1)

    source_embeds = entity_embed[pos_triples[:, 0]]
    relation_embeds = relation_embed[pos_triples[:, 1]]
    tail_embeds = entity_embed[pos_triples[:, 2]]

    x = source_embeds + relation_embeds - tail_embeds
    pos_norm = torch.norm(x, p=2, dim=1)

    source_embeds = entity_embed[neg_triples[:, 0]]
    relation_embeds = relation_embed[neg_triples[:, 1]]
    tail_embeds = entity_embed[neg_triples[:, 2]]

    x = source_embeds + relation_embeds - tail_embeds
    neg_norm = torch.norm(x, p=2, dim=1)

    torch_device = torch.device("cuda:0" if args.cuda else "cpu")
    y = torch.ones(int(args.valid_invalid_ratio)
                   * len_pos_triples, device=torch_device)
    loss = gat_loss_func(pos_norm, neg_norm, y)
    return loss


def render_model_graph(model, Corpus_, train_indices, relation_adj, averaged_entity_vectors):
    graph = make_dot(model(Corpus_.train_adj_matrix, train_indices, relation_adj, averaged_entity_vectors),
                     params=dict(model.named_parameters()))
    graph.render()


def print_grads(model):
    print(model.relation_embed.weight.grad)
    print(model.relation_gat_1.attention_0.a.grad)
    print(model.convKB.fc_layer.weight.grad)
    for name, param in model.named_parameters():
        print(name, param.grad)


def clip_gradients(model, gradient_clip_norm):
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, "norm before clipping is -> ", param.grad.norm())
            torch.nn.utils.clip_grad_norm_(param, args.gradient_clip_norm)
            print(name, "norm beafterfore clipping is -> ", param.grad.norm())


def plot_grad_flow(named_parameters, parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []

    for n, p in zip(named_parameters, parameters):
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="r")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="g")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="r", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="g", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig('initial.png')
    plt.close()


def plot_grad_flow_low(named_parameters, parameters):
    ave_grads = []
    layers = []
    for n, p in zip(named_parameters, parameters):
        # print(n)
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig('initial.png')
    plt.close()
