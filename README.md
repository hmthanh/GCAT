# GCAT (Graph Collaborate Attention Network) - UnDone
![Status](https://img.shields.io/github/issues/hmthanh/GCAT) ![Fork](https://img.shields.io/github/forks/hmthanh/GCAT)
![Stars](https://img.shields.io/github/stars/hmthanh/GCAT)
![License](https://img.shields.io/github/license/hmthanh/GCAT)


Graph Collaborate Attention Network

## Experiment Result

|   GCAT    | H@1   | H@10  |   MR  |  MRR  |
|-----------|:-----:|:-----:|:-----:|:-----:|
| FB15K     | 70.08 | 91.64 |   38  | 0.784 |
| FB15k-237 | 36.06 | 58.32 |  211  | 0.435 |
| WN18RR    | 35.12 | 57.01 | 1974  | 0.430 |

## Structure

```
Root
├── data
│   └── {dataset*} // Dataset
│   │   ├── train.txt
│   │   ├── test.txt
│   │   └── valid.txt
├── output
│   ├── {dataset*} // Result training of each dataset
│   │   ├── WN18RR_cuda_gat_3599.pt ~ "{dataset}_{device}_{model-name}_{last-epoch}"
│   │   └── WN18RR_cuda_result.txt
├── config.json # Config for traning
└── *.py # Source code
└── README.md
```


## Installation

Public Colab : https://drive.google.com/file/d/1uVd_w6vE5C70rmgKLI7BvnhCWegXTMhk/view?usp=sharing

#### Requirements

Using Google Colab with :

* Python `>= 3.6x`
* Pytorch `>= 1.x`

#### Clone
```sh
git clone https://github.com/hmthanh/GCAT.git
```

#### Config

All config store in `config.json` file
```
"dataset": "WN18RR", # Dataset
"data_folder": "./data",
"output_folder": "./output",
"save_gdrive": false, # Use Google Drive to save object
"drive_folder": "/content/drive/My Drive",
"cuda": false, # Use GPU to training
"epochs_gat": 1,
"epochs_conv": 1,
"weight_decay_gat": 5e-06,
"weight_decay_conv": 1e-05,
"pretrained_emb": false,
"embedding_size": 50,
"lr": 0.001,
"get_2hop": true,
"use_2hop": true,
"partial_2hop": false,
"batch_size_gat": 86835,
"valid_invalid_ratio_gat": 2,
"drop_GAT": 0.3,
"alpha": 0.2,
"entity_out_dim": [100, 200],
"nheads_GAT": [2, 2],
"margin": 5,
"batch_size_conv": 128,
"alpha_conv": 0.2,
"valid_invalid_ratio_conv": 40,
"out_channels": 500,
"drop_conv": 0.0
```

#### Run

Because we use Google Colab for training, if you training with larger device, just run `python main.py`

* Step 1 : Create corpus

```sh
python 1_create_corpus.py
```
* Step 2 : Training embedding

```sh
python 2_training_encoder.py
```

* Step 3 : Training prediction

```sh
python 3_training_decoder.py
```

* Step 4 : Evaluation

```sh
python 4_evalution.py
```

## Dataset

* FB15k (Free Base)
* FB15k-237
* WN18 (Word Net)
* WN18RR

## Contact

Email : hmthanhgm@gmail.com | phanminhtam247@gmail.com

## License

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)
- **[MIT license](http://opensource.org/licenses/mit-license.php)**
- Copyright 2020 © <a href="http://hmthanh.github.io" target="_blank">Minh-Thanh Hoang</a>.

*GCAT was modify from KBGAT repos (https://github.com/deepakn97/relationPrediction )*