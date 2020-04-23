import numpy as np
import cv2
from PIL import Image
import argparse
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import transforms as T


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


def predict(img, threshold, model):
    """
        Function to predict and detect objects in frame
        
        params:
            img: pytorch tensor [B, C, H, W]
            threshold: int
            model: pytorch model
        
        returns:
            output: list of 3 tuples of attributes of objects 
                    that have conf_score > threshold
    """
    assert torch.is_tensor(img), "image should be a tensor"
    assert len(img.shape) == 4 and img.shape[1] == 3, "Something is wrong"

    with torch.no_grad():
        predictions = model(img)
        
        # scores of predicted objects
        pred_scores = predictions[0]['scores'].cpu().numpy()
        
        # bounding boxes
        pred_boxes = predictions[0]['boxes'].cpu().numpy()
        boxes = [[(i[0], i[1]), (i[2], i[3])] for i in pred_boxes]
        del pred_boxes
        
        # labels of predicted objects
        pred_classes = predictions[0]['labels'].cpu().numpy()
        labels = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in pred_classes]
        del pred_classes

        output = []
        unique = set()
        for i, val in enumerate(pred_scores):
            if val >= threshold:
                output.append((boxes[i][0], boxes[i][1], labels[i], val))
                unique.add(labels[i])

        return output, unique
    
def load_model():
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model = model.cuda()
    model.eval()

    return model

def process_image(img, resize):
    img = Image.fromarray(img)
    trfs = T.Compose([
        T.Resize((resize, resize)),
        T.ToTensor()
    ])
    img = trfs(img)
    img = img.cuda().unsqueeze(0)
    return img

def recognize(vid_path, file_path, threshold=0.4):
    model = load_model()
    
    # initialize the video read and write objects
    cap = cv2.VideoCapture(vid_path)
    W, H = cap.get(3), cap.get(4) # width and height of frame
    fps = cap.get(cv2.CAP_PROP_FPS) # fps of input video
    print(fps)
    out = cv2.VideoWriter(file_path, cv2.VideoWriter_fourcc('M','J','P','G'), 120,(int(W), int(H)))
    if W > H:
        resize = W
    else:
        resize = H
    
    resize = int(resize)
    while True:
        buffer, frame = cap.read()
        if buffer == False:
            break
        
        # predict the objects
        img = process_image(frame, int(resize))
        outputs, labels = predict(img, threshold, model)
        
        # prepare color map
        colors = np.random.randint(0, 256, size = (len(labels), 3))
        color_map = dict()
        for i, val in enumerate(list(labels)):
            (R, G, B) = colors[i]
            color_map[val] = [int(R), int(G), int(B)]
        
        # Write to the video
        frame = cv2.resize(frame, (resize, resize))
        for i in outputs:
            (ptr0, ptr1, label, _) = i
            (R, G, B) = color_map[label]
            cv2.rectangle(frame, ptr0, ptr1, color_map[label], thickness=2)
            cv2.putText(frame, label, (int(ptr0[0]), int(ptr0[1]-2)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=[255, 255, 255], fontScale=1, thickness=2)
            out.write(cv2.resize(frame, (int(W), int(H))))
            
        

def main(vid_path, file_path, threshold):
    recognize(vid_path, file_path, threshold)
    print("=========Done==========")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect objects in videos")
    parser.add_argument("-i", "--input", required=True, type=str, help="Input video path")
    parser.add_argument("-o", "--output", required=True, type=str, help="Output video path")
    parser.add_argument("-t", "--threshold", default=0.4, type=float)

    args = parser.parse_args()

    main(args.input, args.output, float(args.threshold))
