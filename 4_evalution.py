import torch
from models import SpKBGATConvOnly
from create_config import Config
from utils import load_object

args = Config()
args.load_config()
device = "cuda" if args.cuda else "cpu"

print("Loading corpus")
Corpus_ = load_object(output=args.data_folder, name="corpus")
entity_embeddings = load_object(
    output=args.data_folder, name="entity_embeddings")
relation_embeddings = load_object(
    output=args.data_folder, name="relation_embeddings")

print("Loading model")
model_conv = SpKBGATConvOnly(entity_embeddings, relation_embeddings,
                             args.entity_out_dim, args.entity_out_dim,
                             args.drop_GAT, args.drop_conv, args.alpha, args.alpha_conv,
                             args.nheads_GAT, args.out_channels)
folder = "{output}/{dataset}".format(output=args.output_folder,
                                     dataset=args.dataset)
if args.save_gdrive:
    folder = args.drive_folder
model_name = "{folder}/{dataset}_{device}_{name}_{epoch}.pt".format(
    folder=folder, dataset=args.dataset, device=device, name="conv", epoch=args.epochs_conv - 1)
model_conv.load_state_dict(torch.load(model_name), strict=False)
if args.cuda:
    model_conv.cuda()

model_conv.eval()
with torch.no_grad():
    Corpus_.get_validation_pred(
        args, model_conv, Corpus_.unique_entities_train)

print("4. Evaluation Successfully !")
