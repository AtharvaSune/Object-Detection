from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import os
import argparse
import numpy as np


class FaceDetector(object):

    def __init__(self, image_size=224, device=torch.device("cpu")):
        self.model = MTCNN(keep_all=True, image_size=image_size,
                           device=device, min_face_size=10, thresholds=[0.5, 0.5, 0.5])

    def detect(self, image):
        image = Image.fromarray(image)
        boxes, _ = self.model.detect(image)
        return boxes  # [x1, y1, x2, y2]


class EmbedderGenerator(object):
    def __init__(self, image_size=224, device="cpu", threshold=0.5, model=None, embedding=None, names=None):
        self.embedding_path = embedding
        self.names_path = path
        self.model = cv2.dnn.readNetFromTorch(model)
        self.threshold = threshold
        self.face_detector = FaceDetector(image_size, device)
        self.image_size = image_size

    def predict(self, image):
        h, w, _ = image.shape
        image = cv2.resize(image, (self.image_size, self.image_size))
        detections = self.face_detector.detect(image)
        embedding_vector = None

        if (len(detections) > 0):
            for box in detections:
                box = box * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                faceBlob = cv2.dnn.blobFromImage(
                    face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                self.model.setInput(faceBlob)
                embedding_vector = self.model.forward()

        return embedding_vector


class Dataset(object):
    def __init__(self, path):
        self.dirs = os.listdir(root)
        self.imgs = dict()
        for person in self.dirs:
            self.imgs[person] = os.listdir(os.path.join(root, person))

    def names(self):
        for name in self.dirs:
            yield name

    def get_image(self, name):
        for img in self.imgs[name]:
            yield img
