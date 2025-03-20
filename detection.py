import numpy as np
import cv2
from sklearn import svm, neighbors, linear_model
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os

# Function to preprocess the banknote images
def preprocess_image(image_path):
    if not os.path.isfile(image_path):
        print(f"Error: File '{image_path}' does not exist.")
        return None
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to read image '{image_path}'.")
        return None
    resized_image = cv2.resize(image, (100, 100))
    flattened_image = resized_image.flatten()
    return flattened_image

# Function to load dataset
def load_dataset(notes):
    X = []
    y = []
    for note in notes:
        for i in range(1, 8):
            img_path = f'data/{note}/{i}.jpg'  # Assuming images are stored in folders named after the notes
            preprocessed_img = preprocess_image(img_path)
            if preprocessed_img is not None:
                X.append(preprocessed_img)
                y.append(note)
    return np.array(X), np.array(y)

# Function to train SVM model
def train_svm(X_train, y_train):
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    return clf

# Function to train KNN model
def train_knn(X_train, y_train):
    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train, y_train)
    return clf


# Function to evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Function for user to select model
def select_model(model_name):
    if model_name.lower() == 'svm':
        return 'SVM'
    elif model_name.lower() == 'knn':
        return 'KNN'
    else:
        print("Invalid model selection.")
        return None


# User interface for selecting model and detecting new banknote image
def detect_banknote(model, image_path):
    preprocessed_img = preprocess_image(image_path)
    if preprocessed_img is not None:
        prediction = model.predict([preprocessed_img])[0]
        return prediction
    else:
        return None

