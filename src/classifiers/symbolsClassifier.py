from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier  # MLP is an NN
from sklearn import svm
import numpy as np
import argparse
import cv2
import os
import random
from features.extractfeatures import *
from sklearn.model_selection import train_test_split
import pickle


def load_dataset(path_to_dataset):
    features = []
    labels = []
    img_filenames = os.listdir(path_to_dataset)

    for i, fn in enumerate(img_filenames):
        label = fn.split('_')
        labels.append(label[0])
        space = int(label[2].split('.')[0])
        path = os.path.join(path_to_dataset, fn)
        img = cv2.imread(path, 0)
        img = (img > 100)*1
        featuresVectors = extractFeatures(img, space)
        features.append(featuresVectors[0])
        # show an update every 1,000 images
        if i > 0 and i % 1000 == 0:
            print("[INFO] processed {}/{}".format(i, len(img_filenames)))

    return features, labels


def run_experiment(path_to_dataset, random_seed, classifiers):

    # Load dataset with extracted features
    print('Loading dataset. This will take time ...')
    features, labels = load_dataset(path_to_dataset)
    print('Finished loading dataset.')

    # Since we don't want to know the performance of our classifier on images it has seen before
    # we are going to withhold some images that we will test the classifier on after training
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.2, random_state=random_seed)

    for model_name, model in classifiers.items():
        print('############## Training', model_name, "##############")
        # Train the model only on the training features
        model.fit(train_features, train_labels)

        # Test the model on images it hasn't seen before
        accuracy = model.score(test_features, test_labels)

        print(model_name, 'accuracy:', accuracy*100, '%')


def trainAndSaveSymbolsModel():

    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)

    classifiers = {
        'SVM': svm.LinearSVC(random_state=random_seed),
    }

    path_to_dataset = 'Symbols'
    run_experiment(path_to_dataset, random_seed, classifiers)
    SVM = classifiers['SVM']
    filename = 'symbols_model.sav'
    pickle.dump(SVM, open(filename, 'wb'))


trainAndSaveSymbolsModel()
