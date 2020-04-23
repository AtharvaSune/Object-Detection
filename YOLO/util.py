from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

def predict_transform(predictions, inp_dim, anchors, num_classes, CUDA=True):
    batch_size = predictions.size(0)
    stride = inp_dim // predictions.size(2)
    grid_size = inp_dim // stride
    bbox_attributes = num_classes + 5 # (X_c, Y_c, Width, Height, Confidence)
    num_anchors = len(anchors)

    predictions = predictions.view(batch_size, bbox_attributes * num_anchors, grid_size*grid_size)
    predictions = predictions.transpose(1, 2).contiguous()
    predictions = predictions.view(batch_size, grid_size*grid_size*anchors, bbox_attributes)

    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    predictions[:, :, 0] = torch.sigmoid(predictions[:, :, 0])
    predictions[:, :, 1] = torch.sigmoid(predictions[:, :, 1])
    predictions[:, :, 4] = torch.sigmoid(predictions[:, :, 4])

    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    predictions[:, :, 2] += x_y_offset

    predictions[:, :, 5: 5+num_classes] = torch.sigmoid((predictions[:, :, 5:5+num_classes]))

    predictions[:, :, 4] *= stride

    return predictions

def write_results(prediction, confidence, num_classes, nms_conf = 0.4):
    # prediction shape = B x 10647 x 85
    # B batch size
    # YOLO output boxes
    # number of classes
    conf = (prediction[..., 4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf

    # Performing Non Max suppression
    # Calculating IOU of boxes
    # In prediction we have the box attributes: centerx, centery, height, width
    # Easier to calculate IOU, with top left corner and bottom right corner
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2]/2)
    box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3]/2)
    box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2]/2)
    box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3]/2)
    prediction[:, :, 4] = box_corner[:, :, 4]

    batch_size = prediction.shape[0]
    for batch in range(batch_size):
        image_pred = prediction[batch]
        max_conf, max_conf_score = torch.max(image_pred[:, 5,5+num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:, ;5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)

        # getting rid of bounding boxes with confidence value below threshold
        non_zero = torch.nonzero(image_pred[:,4])
        try:
            image_pred = image_pred[non_zero.squeeze(), :]
        except:
            continue

        img_classes = unique(image_pred[:, -1])

        for cls in img_classes:
            cls_mask = image_pred *(image_pred[:, -1] == cls).__float__().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()
            image_pred_class = image_pred[class_mask_ind].view(-1, 7)

            conf_sort_index = torch.sort(image_pred_class[:, 4],descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)

            for i in range(idx):
                try:
                    ious = bbox_iou()

def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    return unique_tensor
    