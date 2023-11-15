import torch
from torch.utils.data import Dataset
import gzip
import json

class HumanEvalDataset(Dataset):
    def __init__(self, file_name):
        self.file_name = file_name
        self.data = []
        self.load_data(file_name)

    def load_data(self, filename):
        if filename.endswith(".gz"):
            with open(filename, "rb") as gzfp:
                with gzip.open(gzfp, 'rt') as fp:
                    for line in fp:
                        if any(not x.isspace() for x in line):
                            self.data.append(json.loads(line))
        else:
            with open(filename, "r") as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        self.data.append(json.loads(line))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]