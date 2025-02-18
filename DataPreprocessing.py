import json
import os
import pandas as pd
import numpy as np
from PIL import Image, ImageFilter
import cv2
import matplotlib.pyplot as plt
from torchvision import models
import torchvision
from torchvision.models import Weights


root_path = 'C://Users/AtLaS/Desktop/UntiteledNN/'
rawdata_path = root_path + 'datasets/faces/'
binclassd_path = root_path + 'BinaryClassifierDataset/'


os.chdir(rawdata_path+'man')
numpy = np.eye(2, dtype='uint8')
num = numpy[0]

print(num)

'''
annotation = {}

for i, file in enumerate(os.listdir()):
    if i > 8500:
        break
    Image.open(fp=f'{rawdata_path}man/{file}').save(fp=f'{binclassd_path}/{file}')
    annotation.update({file:0})

os.chdir(rawdata_path+'woman')
for i, file in enumerate(os.listdir()):
    if i > 8500:
        break
    Image.open(fp=f'{rawdata_path}woman/{file}').save(fp=f'{binclassd_path}/{file}')
    annotation.update({file:1})

os.chdir(root_path)
with open('annotation.json', 'w') as fp:
    json.dump(annotation, fp)
'''