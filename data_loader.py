import os
import cv2 as cv
import numpy as np
from sklearn.preprocessing import LabelBinarizer

class DataLoader:
    def __init__(self, directory):
        self.directory = directory

    def load_train_data(self):
        """
        Load training data from subdirectories where each subdirectory is a label (category).
        """
        data = []
        labels = []
        #image_paths = []
        
        categories = os.listdir(self.directory)
        for category in categories:
            path = os.path.join(self.directory, category)
            if os.path.isdir(path):  # Check if it is a subdirectory
                for img in os.listdir(path):
                    img_path = os.path.join(path, img)
                    if os.path.isfile(img_path):  # Check if it's a file
                        im = self.preprocess_images(img_path)
                        data.append(im)
                        labels.append(category)
                        #image_paths.append(img_path)
                        
        labels = self.encode_labels(labels)  # One-hot encode labels
        return np.array(data, dtype="float32"), labels, categories

    def load_test_data(self):
        """
        Load test data from a directory with images only (no labels).
        """
        data = []
        image_paths = []

        for img in os.listdir(self.directory):
            img_path = os.path.join(self.directory, img)
            if os.path.isfile(img_path):  # Check if it's a file
                im = self.preprocess_images(img_path)
                data.append(im)
                image_paths.append(img_path)

        return np.array(data, dtype="float32"), image_paths

    def preprocess_images(self, img_path, img_size=(224, 224)):
        """
        Preprocess a single image: load, resize, and normalize.
        """
        im = cv.imread(img_path)  # Load the image
        im = cv.resize(im, img_size)  # Resize the image
        im = np.array(im) / 255.0  # Normalize the image to [0, 1] range
        return im

    def encode_labels(self, labels):
        """
        Convert text labels to one-hot encoded format.
        """
        lb = LabelBinarizer()
        encoded_labels = lb.fit_transform(labels)
        num_classes = len(lb.classes_)
        return encoded_labels
