import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#from sklearn.preprocessing import LabelEncoder
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.model_selection import train_test_split
import cv2
import os

BATCHSIZE =2

image_path = os.path.join('/mrtstorage/users/chli/data_road/training/image_2')
label_path = os.path.join('/mrtstorage/users/chli/data_road/training/gt_image_2')
os.chdir('/mrtstorage/users/chli/data_road/training/image_2')
number = 0
filenames = os.listdir(image_path)
labels = os.listdir(label_path)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

img_data =np.array([]).reshape((1,-1))

def read_batch(number,img_data):
    for filename in filenames[number : number+BATCHSIZE]:
        img =cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        img = img.resize(384,1248)
        img = np.array(img[np.newaxis,:,:])
        print(img.shape)
        if img_data.size == 0:
            img_data= img
        else:
            img_data = np.concatenate((img_data, img))
        print('img_data {}'.format(type(img_data)))
        print(number)


    number = number + BATCHSIZE
    return  number, img_data


number,img_batchdata = read_batch(number,img_data)
print(number, img_batchdata)

'''
# 获取分类类别总数
classes = len(np.unique(y_train))

#对label进行one-hot编码，必须的
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y_train)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
y_train = onehot_encoder.fit_transform(integer_encoded)

#shuffle
X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=0)


model = Sequential()
model.add(Dense(units=1000, activation='relu', input_dim=784))
model.add(Dense(units=classes, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=128)
score = model.evaluate(X_test, y_test, batch_size=128)
'''