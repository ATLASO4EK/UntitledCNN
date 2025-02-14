import torch.nn as nn
import torch.nn.functional as F

#Объявляю класс НС
class ClassifierCNN(nn.Module):
    def __init__(self):
        super().__init__()
        #in (batch, 3, 256, 256)
        self.layer1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding='same'),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        #out (batch, 32, 128, 128)
        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding='same'),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        #out (batch, 64, 64, 64)
        self.layer3 = nn.Sequential(nn.Conv2d(64, 16, 3, padding='same'),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        #out (batch, 16, 32, 32)
        self.flatten = nn.Flatten() #Вытягиваем в тензор размерностью (16*32*32, 1)
        self.dropout=nn.Dropout()
        self.fc1 = nn.Linear(16*32*32, 1024)
        self.fc2 = nn.Linear(1024, 31)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.flatten(x)
        #x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

class LocalizationCNN(nn.Module):
    def __init__(self):
        super().__init__()
        #in (batch, 3, 256, 256)
        self.layer1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding='same'),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        #out (batch, 32, 128, 128)
        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding='same'),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        #out (batch, 64, 64, 64)
        self.layer3 = nn.Sequential(nn.Conv2d(64, 16, 3, padding='same'),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        #out (batch, 16, 32, 32)
        self.flatten = nn.Flatten() #Вытягиваем в тензор размерностью (16*32*32, 1)
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(16*32*32, 1024)
        self.fc2 = nn.Linear(1024, 4)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.flatten(x)
        #x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x