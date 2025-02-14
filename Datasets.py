import json
import os
import torch
from PIL import Image
import torch.utils.data as data


class ClassifierDataset(data.Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform

        with open(os.path.join(path, 'annotations.json'), 'r') as an:
            self.annotations = json.load(an)

        self.len = len(self.annotations)
        self.files = tuple(self.annotations.keys())
        self.targets = tuple(self.annotations.values())


    def __len__(self):
        return self.len

    def __getitem__(self, item):
        path = os.path.join(self.path, self.files[item])
        target = torch.tensor(self.targets[item], dtype=torch.int64)
        img = Image.open(path).convert('RGB')
        img = img.resize((256, 256))

        if self.transform:
            img = self.transform(img)

        return img, target

class LocalizationDataset(data.Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return self.len

    def __getitem__(self,item):
        pass