from create_config import Config
from create_dataset_files import getID

args = Config()
args.load_config()

getID(folder=args.data_folder)

