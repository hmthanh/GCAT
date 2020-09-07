import json

class Config(object):
    def __init__(self,
                 dataset="WN18RR",
                 data_folder="./data/WN18RR/",
                 cuda=False,
                 output_folder="./output/",
                 save_gdrive=False,
                 drive_folder="/content/drive/My Drive",
                 epochs_gat=3600,
                 epochs_conv=200,
                 weight_decay_gat=float(5e-6),
                 weight_decay_conv=float(1e-5),
                 pretrained_emb=True,
                 embedding_size=50,
                 lr=float(1e-3),
                 get_2hop=True,
                 use_2hop=True,
                 partial_2hop=False,
                 batch_size_gat=86835,
                 valid_invalid_ratio_gat=2,
                 drop_GAT=0.3,
                 alpha=0.2,
                 entity_out_dim=[100, 200],
                 nheads_GAT=[2, 2],
                 margin=5,
                 batch_size_conv=128,
                 alpha_conv=0.2,
                 valid_invalid_ratio_conv=40,
                 out_channels=500,
                 drop_conv=0.0, *args, **kwargs):
        self.dataset = dataset
        self.data_folder = data_folder
        self.output_folder = output_folder
        self.drive_folder = drive_folder
        self.save_gdrive = save_gdrive
        self.cuda = cuda

        self.epochs_gat = epochs_gat
        self.epochs_conv = epochs_conv
        self.weight_decay_gat = weight_decay_gat
        self.weight_decay_conv = weight_decay_conv
        self.pretrained_emb = pretrained_emb
        self.embedding_size = embedding_size
        self.lr = lr
        self.get_2hop = get_2hop
        self.use_2hop = use_2hop
        self.partial_2hop = partial_2hop
        self.output_folder = output_folder

        # Tham số cho mô hình GAT
        self.batch_size_gat = batch_size_gat
        self.valid_invalid_ratio_gat = valid_invalid_ratio_gat
        self.drop_GAT = drop_GAT
        self.alpha = alpha
        self.entity_out_dim = entity_out_dim
        self.nheads_GAT = nheads_GAT
        self.margin = margin

        # Tham số cho mô hình ConvKB
        self.batch_size_conv = batch_size_conv
        self.alpha_conv = alpha_conv
        self.valid_invalid_ratio_conv = valid_invalid_ratio_conv
        self.out_channels = out_channels
        self.drop_conv = drop_conv

    def dumps_config(self, config="config.json"):
        with open(config, 'w', encoding='utf-8') as f:
            json.dump(self.__dict__, f, ensure_ascii=False, indent=4)

    def load_config(self, config="config.json"):
        with open(config, encoding='utf-8') as f:
            self.__dict__ = json.load(f)

    def __repr__(self):
        res = ""
        for key, val in self.__dict__.items():
            res += "{key}: {val}\n".format(key=key, val=val)
        return res

#
# args = Config()
# # args.dumps_config()
# args.load_config()
#
# print(args)
