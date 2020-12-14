import os
import numpy as np
import torch.nn.functional as F


class ParameterManager():
    """
    Its tedious to always send all parameters, unified interface
    where we try to access the parameters
    """
    def __init__(self, config, atari_pp, args, device):
        self.config = config
        self.atari_pp = atari_pp
        self.args = vars(args)
        self.device = device

    def __getattr__(self, key):

        if key == "device":
            return self.device

        try:
            value = self.config.get_value(key)
        except:
            try:
                value = self.atari_pp.get_value(key)
            except:
                try:
                    value = self.args[key]
                except:
                    raise ValueError(f"Key: {key} not found in any configuration!")

        return value


class Configuration():

    def __init__(self, file: str, dictionary: dict()):
        self.file = file
        self.dictionary = dictionary

    def __getattr__(self, key):
        return self.get_value(key)

    def set_value(self, key, value):
        self.dictionary[str(key)] = value

    def get_value(self, key):
        try:
            value = self.dictionary[str(key)]
        except:
            raise ValueError(f"Parameter {key} not found in {self.file}.cfg")

        return value


def load_config(file):

    config = dict()

    with open(os.path.join("config", file + ".cfg")) as f:
        for line in f:
            # exclude comments
            if "#" in line:
                continue

            splits = line.strip().split("=")

            if len(splits) != 2:
                continue

            if "'" in splits[1] or "\"" in splits[1]:
                value = splits[1].strip().replace("'", "").replace("\"", "")
            elif splits[1].strip() == "True":
                value = True
            elif splits[1].strip() == "False":
                value = False
            # also allow scientific notation
            elif "." in splits[1] or "e" in splits[1]:
                value = float(splits[1].strip())
            else:
                value = int(splits[1].replace(" ", "").strip())
            config[splits[0].strip()] = value

    return Configuration(file, config)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def entropy(values):
    probs = F.softmax(values, dim=1).detach().cpu().numpy()
    return -np.sum(probs * np.log(probs))