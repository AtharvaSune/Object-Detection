from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle as pkl
import numpy as np
import cv2
from embedding import EmbedderGenerator
from termcolor import colored


class Classify():

    def __init__(self, classifier='svm'):
        if classifier == "knn":
            self.classifier = KNeighborsClassifier(n_neighbors=4)
            self.classifier_name = classifier
        elif classifier == 'svm':
            self.classifier = SVC(C=1.0, kernel="linear", probability=True)
            self.classifier_name = classifier
        elif classifier == 'centroid':
            self.classifier = classifier

    def train(self, data):
        names = []
        embeddings = []
        for key in data:
            for embedding in data[key].T:
                print(embedding.shape)
                embeddings.append(embedding)
                names.append(key)

        self.le = LabelEncoder()
        labels = self.le.fit_transform(names)
        print(labels)
        self.classifier.fit(embeddings, labels)

        self.save()

        print("Training Done")

    def eval(self, data, test_embedding):
        if self.classifier == 'centroid':
            names = list(data.keys())
            prob = []
            for name in names:
                centroid = np.mean(data[name], axis=-1)
                dist = np.linalg.norm((test_embedding - centroid), 2)
                prob.append(1 - (1/(1+np.exp(-dist))))

            prob = np.array(prob)
            idx = np.argmax(prob)
            print(prob)
            print(idx)
            print(names)
            return names[idx]

        elif self.classifier_name == 'svm':
            try:
                classifier = pkl.load(open("./pickle/classifier", "rb"))
                label_encoder = pkl.load(open("./pickle/label_encoder", "rb"))
            except:
                print(f"{colored('[Error]', 'red')} Could not load model")
                print("exiting")
                exit(0)

            preds = classifier.predict_proba([test_embedding])[0]
            idx = np.argmax(preds)
            proba = preds[idx]
            name = label_encoder.classes_[idx]
            return name

    def save(self):
        f = open("./pickle/classifier", "wb")
        pkl.dump(self.classifier, f)
        f.close()

        f = open("./pickle/label_encoder", "wb")
        pkl.dump(self.le, f)
        f.close()


if __name__ == "__main__":

    import os
    try:
        data = pkl.load(open("./pickle/data", "rb"))
    except:
        print("Error loading file")
        exit(0)

    # path = "./dataset/test/20190718_094418.jpg"
    # classify = Classify()
    # if not (os.path.isfile("classifier")):
    #     classify.train(data)
    # else:
    #     embedder = EmbedderGenerator()
    #     embedding = embedder.predict(cv2.imread(path))
    #     print(embedding.shape)
    #     print(f"This is {classify.eval(None, embedding)}")

    cap = cv2.VideoCapture(0)
    while True:
        grabbed, frame = cap.read()

        if not grabbed:
            print(f"{colored('[Info]', 'blue')} Exiting")
            exit(0)

        classifier = Classify()
        if not os.path.isfile("classifier"):
            classifier.train(data)
        else:
            embedder = EmbedderGenerator()
            embedding = embedder.predict(frame)
            print(f"This is {classify.eval(none, embedding)}")
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF:
                print(f"{colored('[Info]', 'blue')} Exiting")
                exit(0)
