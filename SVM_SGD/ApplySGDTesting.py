import sys
import os
import cv2
import numpy as np
import joblib

label = ['cat', 'dog', 'panda']

width = 32
height = 32

model = joblib.load('SVM_SGD/SGD_SVM.pkl')
print('[INFO] Call some image to test ...')
path = 'SVM_SGD/datasets/animals/cats'
listfiles = os.listdir(path)
for(i, imagefile) in enumerate(listfiles):
    imagepath = path+'/'+imagefile
    image = cv2.imread(imagepath)
    img = image.copy()
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    img = img.reshape(1, width*height*3)
    pred = model.predict(img)
    cv2.putText(image, "Label: {}".format(label[pred[0]]),
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow('Org', image)
    key=cv2.waitKey(1000) & 0xFF
    if key == ord('q'):
        break
