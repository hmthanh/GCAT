import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import torch
from torch import nn, optim


class TransE(nn.Module):
    def __init__(self, entity_len, rel_len, embedding_dim, margin=0.5):
        super(TransE, self).__init__()

        self.entity_embeddings = nn.Embedding(entity_len, embedding_dim, ).cuda()
        self.rel_embeddings = nn.Embedding(rel_len, embedding_dim).cuda()

        embeddings_init_bound = 6 / np.sqrt(embedding_dim)
        nn.init.uniform_(
            self.entity_embeddings.weight.data,
            a=-embeddings_init_bound,
            b=+embeddings_init_bound,
        )

        nn.init.uniform_(
            self.rel_embeddings.weight.data,
            a=-embeddings_init_bound,
            b=+embeddings_init_bound,
        )

        self.criterion = nn.MarginRankingLoss(
            margin=margin
        )

        norms = torch.norm(self.rel_embeddings.weight, p=2, dim=1).data
        self.rel_embeddings.weight.data = self.rel_embeddings.weight.data.div(
            norms.view(rel_len, 1).expand_as(self.rel_embeddings.weight))

    def generate_negative_triplets(self, pos_batch, all_entities):
        current_batch_size = len(pos_batch)
        batch_subjs = pos_batch[:, 0:1]
        batch_relations = pos_batch[:, 2:3]
        batch_objs = pos_batch[:, 1:2]

        num_subj_corrupt = len(pos_batch) // 2
        num_obj_corrupt = len(pos_batch) - num_subj_corrupt
        pos_batch = torch.tensor(pos_batch, dtype=torch.long)

        corrupted_subj_indices = np.random.choice(np.arange(0, all_entities.shape[0]), size=num_subj_corrupt)
        corrupted_subjects = np.reshape(all_entities[corrupted_subj_indices], newshape=(-1, 1))
        corrupted_converted_subjects = np.apply_along_axis(self.entities_to_ids, 1, corrupted_subjects).reshape(
            num_subj_corrupt, 1)
        subject_based_corrupted_triples = np.concatenate(
            [corrupted_converted_subjects, (batch_objs[:num_subj_corrupt]).cpu(),
             (batch_relations[:num_subj_corrupt]).cpu()], axis=1)

        corrupted_obj_indices = np.random.choice(np.arange(0, all_entities.shape[0]), size=num_obj_corrupt)
        corrupted_objects = np.reshape(all_entities[corrupted_obj_indices], newshape=(-1, 1))
        corrupted_converted_objects = np.apply_along_axis(self.entities_to_ids, 1, corrupted_objects).reshape(
            num_obj_corrupt, 1)
        object_based_corrupted_triples = np.concatenate(
            [(batch_subjs[num_subj_corrupt:]).cpu(), corrupted_converted_objects,
             (batch_relations[num_subj_corrupt:]).cpu()], axis=1)
        batch_subjs.cuda()
        batch_relations.cuda()
        batch_objs.cuda()
        neg_batch = np.concatenate([subject_based_corrupted_triples, object_based_corrupted_triples], axis=0)
        neg_batch = torch.tensor(neg_batch, dtype=torch.long).cuda()
        return neg_batch

    def entities_to_ids(self, entities):
        return entity2id[entities[0]]

    def forward(self, pos_batch, neg_batch, entity_len):
        pos_score = self.score_triplets(pos_batch, entity_len)
        neg_score = self.score_triplets(neg_batch, entity_len)

        loss = self.compute_loss(pos_score, neg_score)
        return loss

    def train(self, triplets, all_entities, batchsize=32, epochs=1):
        triplets_len = triplets.shape[0]
        optimiser = optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        loss_hist = []
        for epoch in range(epochs):
            print("Epoch: {} is started.".format(epoch))
            for i in range(0, triplets_len, batchsize):
                # raises error if last batch contains only one element!!
                pos_batch = triplets[i:i + batchsize]
                neg_batch = self.generate_negative_triplets(pos_batch, all_entities)
                optimiser.zero_grad()

                loss = self.forward(pos_batch, neg_batch)
                loss_hist.append(loss)
                print("Calculated loss for iteration {}: {}".format(i, loss))
                loss.backward()
                optimiser.step()

        return loss_hist

    def compute_loss(self, pos_scores, neg_scores):
        y = np.repeat([1], repeats=pos_scores.shape[0])
        y = torch.tensor(y, dtype=torch.float)

        positive_scores = torch.tensor(pos_scores, dtype=torch.float)
        negative_scores = torch.tensor(neg_scores, dtype=torch.float)

        loss = self.criterion(pos_scores.cpu(), neg_scores.cpu(), y)

        return loss

    def split_triplets(self, triplets):
        h = triplets[:, 0:1]
        t = triplets[:, 1:2]
        r = triplets[:, 2:3]
        return h, t, r

    def get_embedding_of_triplets(self, triplets):
        heads, tails, relations = self.split_triplets(triplets)
        # print("SHAPE ",self.entity_embeddings.weight[heads].reshape(heads.shape[0],-1).shape)
        return self.entity_embeddings.weight[heads].reshape(heads.shape[0], -1), self.entity_embeddings.weight[
            tails].reshape(heads.shape[0], -1), self.entity_embeddings.weight[relations].reshape(heads.shape[0], -1)

    def score_triplets(self, triplets, entity_len):
        print(self.entity_embeddings.weight.data)
        norms = torch.norm(self.entity_embeddings.weight, dim=1).data
        self.entity_embeddings.weight.data = self.entity_embeddings.weight.data.div(
            norms.view(entity_len, 1).expand_as(self.entity_embeddings.weight))
        print(self.entity_embeddings.weight.data)
        heads, tails, rels = self.get_embedding_of_triplets(triplets)
        sum_res = heads + rels - tails
        distances = torch.norm(sum_res, p=1, dim=1)
        distances_view = distances.view(size=(-1,))

        return distances_view
