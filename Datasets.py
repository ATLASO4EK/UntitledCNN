import json
import os
from symbol import annassign
import numpy as np
import pandas as pd
import torch
from PIL import Image
import torch.utils.data as data


class ClassifierDataset(data.Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform

        with open(os.path.join(path, 'annotation.json'), 'r') as fp:
            self.annotations = json.load(fp)

        self.len = len(self.annotations)
        self.files = tuple(self.annotations.keys())
        self.targets = tuple(self.annotations.values())


    def __len__(self):
        return self.len
    #np.eye(num_classes, dtype='uint8')[self.targets[item]]
    def __getitem__(self, item):
        path = os.path.join(self.path, self.files[item])
        target = torch.tensor(np.eye(2, dtype='uint8')[self.targets[item]], dtype=torch.float32)
        img = Image.open(path).convert('RGB').resize((224, 224))

        if self.transform:
            img = self.transform(img)

        return img, target

class LocalizerDataset(data.Dataset):
    def __init__(self, path, transform):
        self.path = path
        self.transform = transform

        with open(os.path.join(path, 'annotations.json'), 'r') as fp:
            self.annotations = json.load(fp)

        self.len = len(self.annotations)
        self.files = tuple(self.annotations.keys())
        self.targets = tuple(self.annotations.values())

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        path = os.path.join(self.path, self.files[item])
        target = torch.tensor(self.targets[item], dtype=torch.float32)

        img = Image.open(path).convert('RGB').resize((224, 224))

        if self.transform:
            img = self.transform(img)

        return img, target