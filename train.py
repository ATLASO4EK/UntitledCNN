import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from Datasets import *
from CNN import *
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms.v2 as tfs
from tqdm import tqdm

transfroms = tfs.Compose([tfs.Resize((224,224)), tfs.ToImage(), tfs.ToDtype(torch.float32, scale=True)])



d_train = ClassifierDataset('BinaryClassifierDataset', transform=transfroms)
#d_train = ImageFolder('datasets/faces', transform=transfroms)
train_data = data.DataLoader(d_train, batch_size=32, shuffle=True)

epochs = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cl_model = BinClassCNN()
cl_model.to(device)

cl_optimizer = optim.SGD(params=cl_model.parameters(), lr=0.01, momentum=0.9, nesterov=True, weight_decay=0.0005)
#optimizer = optim.Adam(params=model.parameters(), lr=0.01, weight_decay=0.0001)
cl_loss_func = nn.BCELoss()

#model.load_state_dict(torch.load(f=f'ClassifierModels/c_model53 (3).tar'))
cl_model.train()

for _e in range(epochs):
    loss_mean = 0
    lm_count = 0

    train_tqdm = tqdm(train_data, leave=True)
    for x_train, y_train in train_tqdm:
        predict = cl_model(x_train.to(device))
        loss = cl_loss_func(predict, y_train.to(device))

        cl_optimizer.zero_grad()
        loss.backward()
        cl_optimizer.step()

        lm_count += 1
        loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean
        train_tqdm.set_description(f"Epoch [{_e + 1}/{epochs}], loss_mean={loss_mean:.3f}")

    torch.save(cl_model.state_dict(), f=f'BinaryClassifierModels/c_model{_e}.tar')


lc_model = LocalizerCNN()
lc_model.to(device)
#lc_model.load_state_dict(torch.load(f='LocalizerModels/l_model0.tar'))
#lc_optimizer = optim.SGD(params=lc_model.parameters(), lr=0.001, momentum=0.9, nesterov=True, weight_decay=0.0005)
lc_optimizer = optim.Adam(params=lc_model.parameters(), lr=0.001, weight_decay=0.0005)
lc_loss_func = nn.MSELoss()

lc_d_train = LocalizerDataset('LocalizerDataset', transform=transfroms)
lc_train_data = DataLoader(lc_d_train, batch_size=32, shuffle=True)

for _e in range(epochs):
    loss_mean = 0
    lm_count = 0

    lc_train_tqdm = tqdm(lc_train_data, leave=True)
    for x_train, y_train in lc_train_tqdm:
        predict = lc_model(x_train.to(device))
        loss = lc_loss_func(predict, y_train.to(device))

        lc_optimizer.zero_grad()
        loss.backward()
        lc_optimizer.step()

        lm_count += 1
        loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean
        lc_train_tqdm.set_description(f"Epoch [{_e + 1}/{epochs}], loss_mean={loss_mean:.3f}")

    torch.save(lc_model.state_dict(), f=f'LocalizerModels/l_model_sec{_e}.tar')

