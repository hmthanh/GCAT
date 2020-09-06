import torch

from models import SpKBGATConvOnly
from create_config import Config

args = Config()
args.load_config()

print("Loading corpus")

Corpus_ = torch.load(args.data_folder + "Corpus_torch.pt")
entity_embeddings = torch.load(args.data_folder + "entity_embeddings.pt")
relation_embeddings = torch.load(args.data_folder + "relation_embeddings.pt")
node_neighbors_2hop = Corpus_.node_neighbors_2hop

print("Defining model")
model_conv = SpKBGATConvOnly(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                             args.drop_GAT, args.drop_conv, args.alpha, args.alpha_conv,
                             args.nheads_GAT, args.out_channels)

if args.cuda:
    model_conv.cuda()

model_conv.eval()
with torch.no_grad():
    Corpus_.get_validation_pred(args, model_conv, Corpus_.unique_entities_train)

