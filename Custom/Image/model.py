"""
    In this case we will use pretrained models and fine tune them
    for the custom dataset.
"""

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

"""
    fasterrcnn_resnet50_fpn :   is a pretrained model in the torchvision models 
                                It is an object detection model, with resnet-50 as 
                                it's backbone. If you use pretrained = True, then 
                                it is loaded with pretrained weights. These pretrained
                                weights were obtained when they were trained on COCO Dataset

    FasterRCNNPredictor     :   is Standard classification + bounding box regression layers
                                for Fast R-CNN, that has been implemented and made available 
                                in pytorch
"""
def ObjectDetector(num_classes):

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # To finetune the dataset we replace the classifier with a classifier of
    # our choice. That is we use the convolutional layers as feature extractors
    # and then pass it through a Fully Connected Neural Net as classifier
    # which has output (number of classes) as user defined.

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace pretrained neural net with a new one (FasterRCNNPredictor that we imported)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now do the same for the mask predictor
    in_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features,
                                                        hidden_layer,
                                                        num_classes)

    return model


