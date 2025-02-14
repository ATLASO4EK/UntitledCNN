from torch.utils.data import DataLoader

from Datasets import ClassifierDataset
from CNN import ClassifierCNN
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as tfs
from tqdm import tqdm

transfroms = tfs.Compose([tfs.Resize((256, 256)), tfs.ToTensor()])

d_train = ClassifierDataset('ClassifierDataset', transform=transfroms)
train_data = DataLoader(d_train, batch_size=32, shuffle=True)

epochs = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ClassifierCNN()
model.to(device)

optimizer = optim.Adam(params=model.parameters(), lr=0.001, weight_decay=0.0005)
loss_func = nn.CrossEntropyLoss()

model.train()
for epoch in range(epochs):
    train_tqdm = tqdm(train_data, leave=True)
    for x, y in train_tqdm:
        p = model(x.to(device))
        loss = loss_func(p, y.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    torch.save(model.state_dict(), f=f'ClassifierModels/c_model{epoch}.tar')