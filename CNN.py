import torch.nn as nn
import torch.nn.functional as F

#Объявляю класс НС
class BinClassCNN(nn.Module):
    def __init__(self):
        super().__init__()
        #in (batch, 3, 256, 256)
        self.layer1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding='same'),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        #out (batch, 64, 128, 128)
        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding='same'),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        #out (batch, 128, 64, 64)
        self.layer3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding='same'),
                                    nn.ReLU(),
                                    nn.Conv2d(256, 256, 3, padding='same'),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        # out (batch, 256, 32, 32)
        self.layer4 = nn.Sequential(nn.Conv2d(256, 512, 3, padding='same'),
                                    nn.ReLU(),
                                    nn.Conv2d(512, 512, 3, padding='same'),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        # out (batch, 512, 16, 16)
        self.layer5 = nn.Sequential(nn.Conv2d(512, 512, 3, padding='same'),
                                    nn.ReLU(),
                                    nn.Conv2d(512, 512, 3, padding='same'),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        # out (batch, 512, 8, 8)
        self.adpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax()
        #self.dropout=nn.Dropout()
        self.fc1 = nn.Sequential(nn.Linear(512*7*7, 4096, bias=True),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(4096, 4096, bias=True),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(4096, 2, bias=True),
                                 nn.Softmax(1)
                                 )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.adpool(x)
        x = self.flatten(x)
        #x = self.dropout(x)
        x = self.fc1(x)
        #x = self.softmax(x)
        return x

class LocalizerCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # in (batch, 3, 256, 256)
        self.layer1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding='same'),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        # out (batch, 64, 128, 128)
        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding='same'),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        # out (batch, 128, 64, 64)
        self.layer3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding='same'),
                                    nn.ReLU(),
                                    nn.Conv2d(256, 256, 3, padding='same'),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        # out (batch, 256, 32, 32)
        self.layer4 = nn.Sequential(nn.Conv2d(256, 512, 3, padding='same'),
                                    nn.ReLU(),
                                    nn.Conv2d(512, 512, 3, padding='same'),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        # out (batch, 512, 16, 16)
        self.layer5 = nn.Sequential(nn.Conv2d(512, 512, 3, padding='same'),
                                    nn.ReLU(),
                                    nn.Conv2d(512, 512, 3, padding='same'),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        # out (batch, 512, 8, 8)
        self.adpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax()
        # self.dropout=nn.Dropout()
        self.fc1 = nn.Sequential(nn.Linear(512 * 7 * 7, 4096, bias=True),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(4096, 4096, bias=True),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(4096, 4, bias=True),
                                 #nn.Softmax()
                                 )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.adpool(x)
        x = self.flatten(x)
        # x = self.dropout(x)
        x = self.fc1(x)
        # x = self.softmax(x)
        return x
