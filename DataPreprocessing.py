import json
import os
import shutil

import pandas as pd
from PIL import Image, ImageFilter

path = 'C://Users/AtLaS/Desktop/UntiteledNN/archive-3/Original Images/Original Images'
os.chdir(path)

#mean imgs = 100, min - 90 max - 110
#annotations = pd.read_csv('archive-3/Dataset.csv')

#print(len(set(annotations['label'])))

