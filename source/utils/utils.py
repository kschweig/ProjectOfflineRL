import os


class Configuration():

    def __init__(self, file: str, dictionary: dict()):
        self.file = file
        self.dictionary = dictionary

    def __getattr__(self, key):

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
                value = True
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