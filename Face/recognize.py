from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle


class Classify():

    def __init__(self, classifier='knn'):
        if classifier == "knn":
            self.classifier = KNeighborsClassifier(n_neighbors=4)
            self.classifier_name = classifier
        elif classifier == 'svm':
            self.classifier = SVC(kernel="linear", probability=True)
            self.classifier_name = classifier
        elif classifier = 'random_forest':
            self.classifier = RandomForestClassifier()
            self.classifier_name = classifier
        elif classifier = 'centroid'
        self.classifier = classifier

    def train(self, X, y):
        if self.classifier != 'centroid':
            self.classifier.fit(X, y)
            self.save()
        else:
            names = list(set(y))
            lb = LabelEncoder()
            keys = lb.fit_transform(names)

            for i in keys:
                for vec, name in zip(X, y):

    def save(self):
        if self.classifier == 'centroid':

        else:
            name = f"{self.classifier_name}_classifier"
            f = open(name, "w+")
            f.close()

            with open(name, "wb") as f:
                pickle.dump(self.classifier, f)
                f.close()
