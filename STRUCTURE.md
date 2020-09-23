# Graph Embedding in Knowledge Graph

TransE (initiation) -> KBGAT (encoder) -> ConvKB (decoder) -> Evaluation

## Knowlege Graph
Example

()

* `train.txt`
```python
Melania_Trump wife_of Donald_Trump
Donald_Trump president_of U.S
Jeff_Bezos richest_of U.S
Tom_Cruise born_in New_York
New_York state_of U.S
Tesla_Inc founded_in U.S
Melania_Trump first_lady U.S
Tom_Cruise native_of U.S
Thanh friend_of Melania_Trump
```
* `entity2id.txt`
```python
Melania_Trump	0
Donald_Trump	1
U.S				2
Jeff_Bezos		3
Tom_Cruise		4
New_York		5
Tesla_Inc		6
Thanh			7
```

* `relation2id.txt`
```python
wife_of			0
president_of	1
richest_of		2
born_in			3
state_of		4
founded_in		5
first_lady		6
native_of		7
friend_of		8
```

->
```python
[0 0 1] 
[1 1 2] 
[3 2 2] 
[4 3 5] 
[5 4 2] 
[6 5 2] 
[0 6 2] 
[4 7 2] 
[7 8 0]
```

-->

```python
tensor([[5, 4, 2],
        [4, 7, 2],
        [4, 3, 5],
        [1, 1, 2],
        [0, 0, 1],
        [0, 6, 2],
        [7, 8, 0],
        [3, 2, 2],
        [6, 5, 2],
        [1, 4, 2],
        [0, 7, 2],
        [6, 3, 5],
        [2, 1, 2],
        [7, 0, 1],
        [6, 6, 2],
        [1, 8, 0],
        [7, 2, 2],
        [1, 5, 2],
        [5, 4, 1],
        [4, 7, 4],
        [4, 3, 2],
        [1, 1, 7],
        [0, 0, 4],
        [0, 6, 0],
        [7, 8, 6],
        [3, 2, 5],
        [6, 5, 6]])
```



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