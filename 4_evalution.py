import torch
from models import SpKBGATConvOnly
from create_config import Config
from utils import load_object

args = Config()
args.load_config()
device = torch.device("cuda:0" if args.cuda else "cpu")

print("Loading corpus")
Corpus_ = load_object(output=args.data_folder, name="corpus")
entity_embeddings = load_object(output=args.data_folder, name="entity_embeddings")
relation_embeddings = load_object(output=args.data_folder, name="relation_embeddings")
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
