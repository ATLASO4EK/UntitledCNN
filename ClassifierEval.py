import torch
import torchvision.transforms.v2
from PIL import Image
from torch.utils.data import DataLoader

import Datasets
from CNN import ClassifierCNN
import torchvision.transforms as tfs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ClassifierCNN()
model.to(device)

model.load_state_dict(torch.load(f=f'ClassifierModels/c_model4.tar'))

model.eval()
transform = tfs.Compose([tfs.ToTensor()])
d_test = Datasets.ClassifierDataset('test', transform=transform)
test_data = DataLoader(d_test, batch_size=1, shuffle=False)
for x, y in test_data:
    with torch.no_grad():
        p = model(x.to(device))
        p = torch.argmax(p).item()
        ans = y.to(device).item()
print(p, ans)
