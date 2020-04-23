import torch
from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, root, transforms, shuffle=True):
        self.root = root
        self.transform = transforms
        self.images = list(sorted(os.listdir(os.path.join(root, "Images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks"))))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # load image and mask/bbox attrs
        img_path = os.path.join(self.root, "Images", self.images[idx])
        mask_path = os.path.join(self.root, "Images", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = np.array(mask)
        # since this is segmentation / masking, we dont need to convert this
        # image to RGB, as 0 denotes background, and other numbers indicate
        # instances

        # number of unique instances
        obj_ids = np.unique(mask)
        # first_id is the background, which we are not interested in
        obj_ids = obj_ids[1:]

        masks = mask == obj_ids[:, None, None]

        # bounding box coordinates for the masks
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            x_min = np.min(pos[1])
            x_max = np.max(pos[1])
            y_min = np.min(pos[0])
            y_max = np.max(pos[0])
            boxes.append([x_min, y_min, x_max, y_max])
        
        # in case of object detection, directly read the file that provides
        # box coordinates, labels for each box, and remember to convert the 
        # coordinates proprtionally.
        # Example: img 600x600, centerx = 200, centery = 300, width = 100, height = 150
        # then proportionally centerx = 0.3333, centery = 0.5, width = 1/6, height = 0.25

        # convert everthing to tensor
        boxes = torch.as_tensor(boxes, torch.float32)
        labels = torch.ones((num_objs, ), torch.int64) # in this dataset there is only one class
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        target = {}
        target['boxes'] = boxes
        target['masks'] = masks
        target['labels'] = labels
        target['image_id'] = torch.as_tensor([idx])
        
        # area of box is best measure to calculate IOU/Jaccard Index of two boxes
        # area of box = ((ymax - ymin) * (xmax - xmin))
        boxes_area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target['area'] = boxes_area

        # In segmentation you need to keep another variable, in case the complete image
        # is just a background, that is represented by all zeros.
        iscrowd = torch.zeros((num_objs, ), torch.int64)
        target['crowd'] = iscrowd
        
        # apply transformations if any
        if self.transform is not None:
            img, target = self.transform(img, target)
        
        return img, target