import torch
from torchvision.transforms import transforms as T
import numpy as np
import torchvision
from dataset import CustomDataset
from model import ObjectDetector

def train(root, num_epochs, num_classes, gpu=None, batch_size = 32):
    trfs = T.Compose([
        T.ToTensor(),
        T.RandomHorizontalFlip(0.5)
    ])

    raw_dataset = CustomDataset(root, trfs, False)
    dataloader = torch.utils.data.DataLoader(raw_dataset, 
                                            batch_size,
                                            shuffle = True)

    model = ObjectDetector(num_classes)
    
    if gpu is not None:
        model = model.cuda()
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, 0.01, 0.95, weight_decay=0.0005)

    for _ in range(num_epochs):
        
        for _, (image, targets) in enumerate(dataloader):
            if gpu is not None:
                image = image.cuda()
                target = [{k:v.cuda() for k, v in t.items()} for t in targets]
                
            loss_dict = model(image, target,)

            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

    path = "./MaskRCNN-FineTune.pth"
    save_model(model, path)

def save_model(model, path):
    torch.save(model.state_dict(), path)