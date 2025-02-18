import torch
import torchvision.transforms.v2 as tfs
from CNN import BinClassCNN, LocalizerCNN
from PIL import Image

def get_face_class(img_path, classifier_model, localizer_model):
    img = Image.open(fp=img_path).convert('RGB').resize((256, 256))
    transform = tfs.Compose([tfs.ToImage(), tfs.ToDtype(torch.float32, scale=True)])
    img_transformed = transform(img)
    cl_pr = torch.argmax(classifier_model(img_transformed)).item()
    x, y, w, h = localizer_model(img_transformed)
    face_img = img.crop((x * 256, y*256, w*256, h*256))

    return cl_pr, face_img

def load_model_state_dicts(classifier_path, localizer_path):
    classifier_model = BinClassCNN()
    localizer_model = LocalizerCNN()
    classifier_model.load_state_dict(torch.load(fp=classifier_path))
    localizer_model.load_state_dict(torch.load(fp=localizer_path))

    return classifier_model,  localizer_model
