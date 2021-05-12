import sys
import os
import cv2
import numpy as np
import argparse

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib


class DatasetLoad:
    def __init__(self, width=64, height=64, pre_type='Resize'):
        self.width = width
        self.height = height
        self.pre_type = pre_type

    def load(self, pathes, verbose=-1):

        datas = []
        labels = []

        mainfolders = os.listdir(pathes)

        for folder in mainfolders:
            fullpath = os.path.join(
                pathes, folder)
            listfiles = os.listdir(fullpath)

            if verbose > 0:
                print('[INFO] loading', folder, ' ...')

            for(i, imagefile) in enumerate(listfiles):
                imagepath = pathes+'/'+folder+'/'+imagefile
                image = cv2.imread(imagepath)
                label = folder

                if(self.pre_type == 'Resize'):
                    image = cv2.resize(
                        image, (self.width, self.height), interpolation=cv2.INTER_AREA)
                datas.append(image)
                labels.append(label)
                if(verbose > 0 and i > 0 and (i+1) % verbose == 0):
                    print('[INFO] processed {}/{}'.format(i+1, len(listfiles)))
        return (np.array(datas), np.array(labels))


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default="SVM_SGD/datasets/animals",
                help="path to input dataset")
args = vars(ap.parse_args())
pathes = args["dataset"]

width = 32
height = 32

data = DatasetLoad(width, height)

print('[INFO] loading datasets...')

label = ['cat', 'dog', 'panda']

datas, labels = data.load(pathes, verbose=500)
print('[INFO] shape of dates = ', datas.shape)

flat_image = datas.shape[1]*datas.shape[2]*datas.shape[3]
datas = datas.reshape((datas.shape[0], flat_image))
print('[INFO] new datas shape = ', datas.shape)

print('[INFO] split dataset to training and testing dataset ...')
(trainX, testX, trainY, testY) = train_test_split(
    datas, labels, test_size=0.30, random_state=45)

print(trainY)
print(testY)
le = LabelEncoder()
trainY = le.fit_transform(trainY)
testY = le.fit_transform(testY)
print(trainY)
print(testY)

model = SGDClassifier(loss='log', penalty='l2',learning_rate='optimal', eta0=0.01, max_iter=1000)

print("[INFO] training...")
model.fit(trainX, trainY)

print("[INFO] evaluating classifier...")
predictions = model.predict(testX)
print(classification_report(testY, predictions, target_names=le.classes_))

joblib.dump(model,'SGD_SVM.pkl')
