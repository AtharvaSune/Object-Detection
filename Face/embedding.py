from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import os
import argparse
import numpy as np
import torch
from PIL import Image
from termcolor import colored


class FaceDetector(object):

    def __init__(self, image_size=224, device=torch.device("cpu")):
        self.model = MTCNN(keep_all=True, image_size=image_size,
                           device=device, min_face_size=10, thresholds=[0.5, 0.5, 0.5])

    def detect(self, image):
        image = Image.fromarray(image)
        boxes, _ = self.model.detect(image)
        return boxes  # [x1, y1, x2, y2]

    def __call__(self, image):
        image = Image.fromarray(image)
        return self.model(image, return_prob=True)


class EmbedderGenerator(object):
    def __init__(self, image_size=224, device=torch.device("cpu"), threshold=0.5, model=None, embedding=None, names=None):
        self.embedding_path = embedding
        self.names_path = names
        self.embedder = InceptionResnetV1(
            pretrained="vggface2").eval().to(device)
        self.threshold = threshold
        self.face_detector = FaceDetector(image_size, device)
        self.image_size = image_size

    def predict(self, image):
        if image is None:
            return np.zeros((512, ))
        h, w, _ = image.shape
        image = cv2.resize(image, (self.image_size, self.image_size))
        cropped, prob = self.face_detector(image)
        if cropped is None:
            return np.zeros((512, ))
        cropped_D = cropped.cpu().numpy()[0, :, :, :]
        cropped_D = np.transpose(cropped_D, (1, 2, 0))

        embedding = self.embedder(cropped).detach()
        if embedding.size(0) != 1:
            return np.zeros((512, ))
        embedding = torch.squeeze(embedding, axis=0)
        embedding = embedding.cpu().numpy()
        print(f"Embedding: {embedding.shape}")
        return embedding


class Dataset(object):
    def __init__(self, root):
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


if __name__ == "__main__":
    root = "./dataset/train"
    data = Dataset(root)
    embed = EmbedderGenerator(model="./face detection model/nn4.small2.v1.t7")
    save_data = dict()
    for _, name in enumerate(data.names()):
        embedding_vector_list = []
        print(f"Procesing photos of {colored(name, 'green')}")
        n = len(data.imgs[name])
        for _, image in enumerate(data.get_image(name)):
            if _ % 5 == 4:
                print(f"{colored('[INFO]', 'blue')} {_+1}/{n} photos done !!!")
            img = cv2.imread(os.path.join(root, name, image))
            embedding_vector = embed.predict(img)
            embedding_vector_list.append(embedding_vector)

        print(f"shape: {np.array(embedding_vector_list).shape}")
        if name not in save_data.keys():
            print(f"Adding {name}")
            save_data[name] = np.array(embedding_vector_list).T
        del embedding_vector_list

    for key in save_data:
        print(f"{key}: {save_data[key].shape}")

    import pickle as pkl
    try:
        f = open("./pickle/data", 'w+')
        f.close()
    except:
        print(f"{colored('Error opening file', 'red')} !!!")

    with open("./pickle/data", 'wb') as f:
        pkl.dump(save_data, f)
        f.close()
