import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#from sklearn.preprocessing import LabelEncoder
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.model_selection import train_test_split

import os

BATCHSIZE =2

image_path = os.path.join('/mrtstorage/users/chli/data_road/training/image_2')
label_path = os.path.join('/mrtstorage/users/chli/data_road/training/gt_image_2')
#os.chdir('/mrtstorage/users/chli/data_road/training/image_2')
number = 0
filenames = os.listdir(image_path)
labels = os.listdir(label_path)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_png(image_string)
  image_resized = tf.image.resize_images(image_decoded, [1248, 384])
  image_resized.set_shape([1248, 384, 3])

  label_string = tf.read_file(label)
  label_decoded = tf.image.decode_png(label_string, channels =1)
  label_decoded = tf.image.convert_image_dtype(label_decoded, tf.float32)
  label_resized = tf.image.resize_images(label_decoded, [1248, 384])
  label_new = label_resized/255.0


  with tf.Session() as sess:
      print(label_new)
      plt.figure(1)
      plt.imshow(label_new.eval())
      plt.show()



  #print('image_resized {}'.format(image_resized))
  return image_resized, label



# 此时dataset中的一个元素是(filename, label)
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

# 此时dataset中的一个元素是(image_resized, label)
dataset = dataset.map(_parse_function)

# 此时dataset中的一个元素是(image_resized_batch, label_batch)
#dataset = dataset.shuffle(buffer_size=100).batch(32).repeat(10)
#print(dataset)




def read_batch(number):
    for filename in filenames[number : number+BATCHSIZE]:
        img = cv2.imread(filename)
        number = number +  BATCHSIZE
        print(number)
        return  img , number
        break

#read_batch(number)

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