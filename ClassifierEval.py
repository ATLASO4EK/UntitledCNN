import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

import Datasets
from CNN import PersonClassifierCNN
import torchvision.transforms.v2 as tfs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PersonClassifierCNN()
model.to(device)

model.load_state_dict(torch.load(f=f'ClassifierModels/c_model_thd5.tar'))



model.eval()
transform = tfs.Compose([tfs.ToImage(), tfs.ToDtype(torch.float32, scale=True), tfs.Normalize([0.5,0.5,0.5],
                         [0.5,0.5,0.5])])
d_test = Datasets.ClassifierDataset('ClassifierDataset', transform=transform)
test_data = DataLoader(d_test, batch_size=1, shuffle=True)

for x, y in test_data:
    with torch.no_grad():
        p = model(x.to(device))
        p = torch.argmax().item()
        ans = torch.argmax(y.to(device)).item()
        print(p, ans)
