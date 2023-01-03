# khai báo thư viện
import numpy as np
from numpy import *
import cv2
import pandas as pd
from PIL import Image, ImageOps
import pickle 
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from tensorflow.keras import callbacks

# lấy data từ tập dữ liệu có sẵn MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# chuyển đổi lớp vector thành ma trận lớp nhị phân
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#bulid model CNN
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])

# xác định lệnh gọi lại dừng sớm
early_stopping = callbacks.EarlyStopping(patience = 20, restore_best_weights=True, monitor='loss', min_delta=0)
lr_schedule = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=6, min_delta=0.0001)

# Huấn luyện mô hình
# hist = model.fit(x_train, y_train, batch_size = 128, epochs = 30, callbacks = [early_stopping, lr_schedule])
# print("The model has successfully trained")

# Lưu mô hình 
# model.save('mnist.h5')
# print("Saving the model as mnist.h5")

# Đánh giá model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model = keras.models.load_model('mnist.h5')

def predict_image(image_path):
  image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
  image = cv2.threshold(image, 110, 255, cv2.THRESH_BINARY)[1]
  cv2.imshow('số', image)
  image = cv2.resize(image, (28, 28),  interpolation=cv2.INTER_AREA).flatten()
  image = np.array(image).reshape(1, 28, 28, 1)%255
  predict = model.predict(image) 
  print('đây là số : ' , predict)

predict_image('11.jpg')
