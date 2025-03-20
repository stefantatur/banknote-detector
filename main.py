import numpy as np
import cv2
import os
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import shutil
from detection import load_dataset, train_svm, train_knn, evaluate_model, select_model, detect_banknote


# Load dataset
#notes = ['100', '200', '500', '1000', '5000']
def run(model, new_banknote_image_path):
    notes = ['1000', '5000']
    X, y = load_dataset(notes)

    # Split dataset into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models
    svm_model = train_svm(X_train, y_train)
    knn_model = train_knn(X_train, y_train)

    # Evaluate models
    svm_accuracy = evaluate_model(svm_model, X_test, y_test)
    knn_accuracy = evaluate_model(knn_model, X_test, y_test)

    print(f"SVM Accuracy: {svm_accuracy}")
    print(f"KNN Accuracy: {knn_accuracy}")

    # Example of using the user interface
    #selected_model = input("Select a model (SVM, KNN): ")
    selected_model = select_model(model)

    if selected_model:
        #new_banknote_image_path = input("Enter the path of the new banknote image: ")
        #new_banknote_image_path = r"C:\Users\steph\PycharmProjects\detect_banknotes\sample_2.jpg"
        prediction = detect_banknote(svm_model, new_banknote_image_path)
        if prediction is not None:
            return prediction

