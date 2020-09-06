import torch
from models import SpKBGATConvOnly
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
model_conv = SpKBGATConvOnly(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                             args.drop_GAT, args.drop_conv, args.alpha, args.alpha_conv,
                             args.nheads_GAT, args.out_channels)

if args.cuda:
    model_conv.cuda()

model_conv.eval()
with torch.no_grad():
    Corpus_.get_validation_pred(args, model_conv, Corpus_.unique_entities_train)

print("4. Evaluation Successfully !")
