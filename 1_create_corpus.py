import torch

import numpy as np

from preprocess import  init_embeddings, build_data
from create_batch import Corpus
import os

from create_config import Config

args = Config()
args.load_config()

train_data, validation_data, test_data, entity2id, relation2id, headTailSelector, unique_entities_train = build_data(args.data_folder, is_unweigted=False, directed=True)
(test_triples, test_adjacency_mat) = test_data
print("test_triples", np.array(test_triples).shape)
print("test_adjacency_mat", np.array(test_adjacency_mat).shape)
print("entity2id", len(entity2id))
print("relation2id", len(relation2id))
print("headTailSelector", len(headTailSelector))
print("unique_entities_train", np.array(unique_entities_train).shape, "\n")


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

Corpus_ = Corpus(args, train_data, validation_data, test_data, entity2id, relation2id, headTailSelector,
                args.batch_size_gat, args.valid_invalid_ratio_gat, unique_entities_train, args.get_2hop)

entity_embeddings = torch.FloatTensor(entity_embeddings)
relation_embeddings = torch.FloatTensor(relation_embeddings)

device = "gpu" if args.cuda else "cpu"
if (args.save_gdrive):
    torch.save(Corpus_, "{output}corpus_{device}.pt".format(output=args.drive_folder, device=device))
    torch.save(entity_embeddings, "{output}entity_embeddings_{device}.pt".format(output=args.drive_folder, device=device))
    torch.save(relation_embeddings, "{output}relation_embeddings_{device}.pt".format(output=args.drive_folder, device=device))
    node_neighbors_2hop = Corpus_.node_neighbors_2hop
else:
    torch.save(Corpus_, "{output}corpus_{device}.pt".format(output=args.data_folder, device=device))
    torch.save(entity_embeddings, "{output}entity_embeddings_{device}.pt".format(output=args.data_folder, device=device))
    torch.save(relation_embeddings, "{output}relation_embeddings_{device}.pt".format(output=args.data_folder, device=device))
    node_neighbors_2hop = Corpus_.node_neighbors_2hop

print("1. Created Corpus Successfully !")
