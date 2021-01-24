from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import cv2
import os
import random
from skimage.morphology import skeletonize
from sklearn.model_selection import train_test_split
import pickle

target_img_size = (32, 32)

def extract_hog_features(img):
    img = cv2.resize(img, target_img_size)
    win_size = (32, 32)
    cell_size = (4, 4)
    block_size_in_cells = (2, 2)

    block_size = (block_size_in_cells[1] * cell_size[1], block_size_in_cells[0] * cell_size[0])
    block_stride = (cell_size[1], cell_size[0])
    nbins = 9  # Number of orientation bins
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    h = hog.compute(img)
    h = h.flatten()
    return h.flatten()

def extract_features(img):
    return extract_hog_features(img)


def load_dataset(path_to_dataset):
    features = []
    labels = []
    img_filenames = os.listdir(path_to_dataset)
    for i, fn in enumerate(img_filenames):
        if fn.split('.')[-1] != 'jpg':
            continue

        label = fn.split('.')[0]
        labels.append(label)

        path = os.path.join(path_to_dataset, fn)
        img = cv2.imread(path, 0)
        threshold = 200
        binarizedImg = np.zeros(img.shape)
        binarizedImg = np.where(img > threshold, 1, 0)
        ###########################
        skeletonizedImg = skeletonize(binarizedImg)
        channels = np.zeros((img.shape[0], img.shape[1], 3)).astype('uint8')
        channels[:, :, 0] = skeletonizedImg
        channels[:, :, 1] = skeletonizedImg
        channels[:, :, 2] = skeletonizedImg

        features.append(extract_features(channels))

        # show an update every 1,000 images
        if i > 0 and i % 1000 == 0:
            print("[INFO] processed {}/{}".format(i, len(img_filenames)))

    return features, labels

def run_experiment(path_to_dataset,classifiers,random_seed):
    # Load dataset with extracted features
    print('Loading dataset. This will take time ...')
    features, labels = load_dataset(path_to_dataset)
    print('Finished loading dataset.')

    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.2, random_state=random_seed)

    for model_name, model in classifiers.items():
        print('############## Training', model_name, "##############")
        # Train the model only on the training features
        model.fit(train_features, train_labels)

        # Test the model on images it hasn't seen before
        accuracy = model.score(test_features, test_labels)

        print(model_name, 'accuracy:', accuracy * 100, '%')


def extractDigitsFeatures(img):
    img = 1 - img
    skeletonizedImg = skeletonize(img)
    channels = np.zeros((img.shape[0], img.shape[1], 3)).astype('uint8')
    channels[:, :, 0] = skeletonizedImg
    channels[:, :, 1] = skeletonizedImg
    channels[:, :, 2] = skeletonizedImg
    features = extract_features(channels)
    return features

def digitsClassifier(test_img_path):
    path_to_dataset = r'dataset'
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    classifiers = {
        'KNN': KNeighborsClassifier(n_neighbors=7),
    }
    run_experiment(path_to_dataset,classifiers,random_seed)

    nn = classifiers['KNN']
    filename = 'digits_model.sav'
    pickle.dump(nn, open(filename, 'wb'))
