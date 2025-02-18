from array import array

import numpy
import torch
from PIL import Image
import torchvision.transforms.v2 as tfs
from CNN import *

model = LocalizerCNN()
model.load_state_dict(torch.load(f='LocalizerModels/l_model2.tar'))

transform = tfs.Compose([ tfs.ToImage(), tfs.ToDtype(dtype=torch.float32) ])

img = Image.open('test/Claire Holt_26.jpg').convert('RGB').resize((256,256))

img_tr = transform(img).unsqueeze(0)

with torch.no_grad():
    pr = model(img_tr)
pr = numpy.array(pr)
pr = pr.tolist()

print(len(pr))
img = img.crop((pr[0][0]*256, pr[0][1]*256, pr[0][2]*256, pr[0][3]*256))

img.show()