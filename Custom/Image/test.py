import torch
from PIL import Image
from model import ObjectDetector
from torchvision.transforms import transforms as T

def test(img_path, file_path, num_classes, model_path, gpu):
    model = ObjectDetector(num_classes)
    model.load_state_dict(torch.load(model_path))

    img = Image.open(img_path)
    trfms = T.Compose([
        T.ToTensor()
    ])
    img = trfms(img)
    if gpu is not None:
        img = img.cuda()
        model = model.cuda()
    
    model.eval()
    predictions = model(img)

    return predictions
