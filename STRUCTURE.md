# Graph Embedding in Knowledge Graph

TransE (initiation) -> KBGAT (encoder) -> ConvKB (decoder) -> Evaluation

### TransE


### Model KBGAT (encoder)

Layers

```python
SpKBGATModified(
  (sparse_gat_1): SpGAT(
    (dropout_layer): Dropout(p=0.3, inplace=False)
    (attention_0): SpGraphAttentionLayer (50 -> 100)
    (attention_1): SpGraphAttentionLayer (50 -> 100)
    (out_att): SpGraphAttentionLayer (200 -> 200)
  )
)
```

Architecture
```python
final_entity_embeddings : torch.Size([8, 200])
final_relation_embeddings : torch.Size([9, 200])
entity_embeddings : torch.Size([8, 50])
relation_embeddings : torch.Size([9, 50])
W_entities : torch.Size([50, 200])
sparse_gat_1.W : torch.Size([50, 200])
sparse_gat_1.attention_0.a : torch.Size([100, 150])
sparse_gat_1.attention_0.a_2 : torch.Size([1, 100])
sparse_gat_1.attention_1.a : torch.Size([100, 150])
sparse_gat_1.attention_1.a_2 : torch.Size([1, 100])
sparse_gat_1.out_att.a : torch.Size([200, 600])
sparse_gat_1.out_att.a_2 : torch.Size([1, 200])
```

### Model ConvKB (decoder)

Layers

```python
SpKBGATModified(
  (sparse_gat_1): SpGAT(
    (dropout_layer): Dropout(p=0.3, inplace=False)
    (attention_0): SpGraphAttentionLayer (50 -> 100)
    (attention_1): SpGraphAttentionLayer (50 -> 100)
    (out_att): SpGraphAttentionLayer (200 -> 200)
  )
)
```

Architecture
```
final_entity_embeddings : torch.Size([8, 200])
final_relation_embeddings : torch.Size([9, 200])
convKB.conv_layer.weight : torch.Size([500, 1, 1, 3])
convKB.conv_layer.bias : torch.Size([500])
convKB.fc_layer.weight : torch.Size([1, 100000])
convKB.fc_layer.bias : torch.Size([1])
```