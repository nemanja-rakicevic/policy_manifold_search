
import json
import pickle


def load_metadata(datapath):
    """Extract class experiment metadata """
    with open(datapath + '/experiment_metadata.json', 'r') as f:
        args_dict = json.load(f)
    return args_dict


def load_dataset(datapath):
    """Extract class label info """
    with open(datapath + "/experiment_dataset.dat", "rb") as f:
        data_dict = pickle.load(f)
    return data_dict


def save_metadata(file, datapath):
    with open(datapath + "/experiment_metadata.json", 'w') as outfile:
        json.dump(file, outfile, sort_keys=True, indent=4)
