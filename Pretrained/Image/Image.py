import cv2
import torch
import numpy as np
import argparse
from PIL import Image
from torchvision.transforms import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def predict(img_path, threshold):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model = model.cuda()
    model.eval()


    try:
        img = Image.open(img_path)
    except:
        print("Image not found......exiting")
        exit(0)
    trf = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor()
    ])
    img_ = trf(img)
    img_ = img_.cuda()
    img_ = img_.unsqueeze(0)
    with torch.no_grad():
        predictions = model(img_)
        pred_scores = predictions[0]['scores'].cpu().numpy()
        pred_boxes = predictions[0]['boxes'].cpu().numpy()
        pred_classes = predictions[0]['labels'].cpu().numpy()
        pred_labels = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in pred_classes]
        del pred_classes
        output = []
        for i, val in enumerate(pred_scores):
            if val >= threshold:
                output.append(((pred_boxes[i][0], pred_boxes[i][1]), (pred_boxes[i][2], pred_boxes[i][3]), pred_labels[i], val))

        return output

def recognize(img_path, filepath, threshold):
    outputs = predict(img_path, threshold)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (1024, 1024))
    unique = set()
    for i in outputs:
        unique.add(i[2])
    colors = np.random.randint(0, 256, size = (len(unique), 3))
    color_map = dict()
    for i, val in enumerate(list(unique)):
        (R, G, B) = colors[i]
        color_map[val] = [R, G, B]
    
    for i in outputs:
        (ptr0, ptr1, label, score) = i
        score = np.round(score, 2)
        (R, G, B) = color_map[label]
        text = label+ " "+str(score)
        cv2.rectangle(img, ptr0, ptr1, [int(R), int(G), int(B)], thickness=2)
        
        if len(color_map) > 4:
            if score > 0.9:
                cv2.putText(img, text, (int(ptr0[0]), int(ptr0[1]-2)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=[255, 255, 255], fontScale=1, thickness=2)
        else:
            thickness = 2
            cv2.putText(img, text, (int(ptr0[0]), int(ptr0[1]-2)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=[255, 255, 255], fontScale=1, thickness=thickness)
    
    cv2.imwrite(filepath, cv2.resize(img, (1920, 1080)))
    print("=========Done==========")

def main(img_path, filepath, threshold = 0.4):
    recognize(img_path, filepath, threshold)
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect Objects in images available in COCO Dataset")
    parser.add_argument("-i", '--input', required=True, help="path to input image", type=str)
    parser.add_argument("-o",'--output', required=True, help="path to output", type=str)

    args = parser.parse_args()

    main(args.input, args.output)