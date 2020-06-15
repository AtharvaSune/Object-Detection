import cv2
import os
import argparse

def write_video(img, box, out):
    # cv2.rectangle(img, box[0], box[1], [255, 255, 255], thickness=1)
    out.write(img)

def convert_to_video(root, box):
    images = os.scandir(root)
    W = 360
    H = 240
    out = cv2.VideoWriter("output2.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 20,(int(W), int(H)))
    boxes = open(box, 'r').read().split("\n")
    for i, val in enumerate(boxes[:-1]):
        val = val.split("\t")
        val = [int(j) for j in val]
        boxes[i] = [(val[0], val[1]), (val[0]+val[2], val[1]+val[3])]
    file_name = []
    for i in images:
        file_name.append(i.name)
    file_name.sort()
    for i, val in enumerate(file_name):
        img = cv2.imread(os.path.join(root, val))
        write_video(img, boxes[i], out)    
    
    print("===========Done============")
    exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="stitch photos to form video")
    parser.add_argument("-r", "--root", required=True, type=str, help="directory path to the images")
    parser.add_argument("-b", "--box", required=True, type=str, help="directory path to box file")
    args = parser.parse_args()

    convert_to_video(args.root, args.box)