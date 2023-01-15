from skimage.io import imread
from skimage.transform import resize
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPool2D
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import os
from tqdm import tqdm
import csv


imgSize = 224
dic = {0: 'Basketball',
       1: 'Football',
       2: 'Rowing',
       3: 'Swimming',
       4: 'Tennis',
       5: 'Yoga'}

trainDataPath = "Train/"
testDataPath = "Test/"
modelName = 'sportsCNN'

# To increase our DataSet
def dataAugmentation():
    dataGen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest")

    dataPath = "Train/"
    for imgName in tqdm(os.listdir(dataPath)):
        print('\n', imgName[:-4])
        path = os.path.join(dataPath, imgName)
        image = cv2.imread(path, 1)
        print(np.array(image).shape)
        img = np.array(image)
        img = img.reshape((1,) + img.shape)
        i = 0
        for batch in dataGen.flow(img, batch_size=1,
                                  save_to_dir='Temp', save_prefix=imgName[:-4], save_format='jpg'):
            i = i + 1
            if i == 5:
                break


def readData():
    xtrain = []
    ytrain = []
    for imgName in tqdm(os.listdir(trainDataPath)):
        path = os.path.join(trainDataPath, imgName)
        image = cv2.imread(path)
        xtrain.append(np.array(cv2.resize(image, (imgSize, imgSize, 3))))
        if imgName[0] == 'B':
            ytrain.append([1, 0, 0, 0, 0, 0])
        elif imgName[0] == 'F':
            ytrain.append([0, 1, 0, 0, 0, 0])
        elif imgName[0] == 'R':
            ytrain.append([0, 0, 1, 0, 0, 0])
        elif imgName[0] == 'S':
            ytrain.append([0, 0, 0, 1, 0, 0])
        elif imgName[0] == 'T':
            ytrain.append([0, 0, 0, 0, 1, 0])
        elif imgName[0] == 'Y':
            ytrain.append([0, 0, 0, 0, 0, 1])

    xtrain = np.array(xtrain)
    xtrain = xtrain.reshape(-1, imgSize, imgSize, 3)
    ytrain = np.array(ytrain)
    x_train, x_test, y_train, y_test = train_test_split(xtrain, ytrain, test_size=0.2, shuffle=True)
    xtrain = []
    ytrain = []
    np.save('x_train.npy', x_train)
    np.save('y_train.npy', y_train)
    np.save('x_test.npy', x_test)
    np.save('y_test.npy', y_test)
    return x_train, x_test, y_train, y_test


# if os.path.exists('x_test.npy'):
#     x_train = np.load('x_train.npy')
#     y_train = np.load('y_train.npy')
#     x_test = np.load('x_test.npy')
#     y_test = np.load('y_test.npy')
# else:
#     x_train, x_test, y_train, y_test = readData()


model = Sequential()
model.add(Conv2D(input_shape=(imgSize,imgSize,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),name='vgg16'))
model.add(Flatten(name='flatten'))
model.add(Dense(256, activation='relu', name='fc1'))
model.add(Dense(128, activation='relu', name='fc2'))
model.add(Dense(6, activation='softmax', name='output'))
model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.0001),metrics=["accuracy"])
model.summary()

model.load_weights('weights.h5')

# if os.path.exists('weights.h5'):
#     model.load_weights('weights.h5')
# else:
#     model.fit(x_train, y_train, batch_size=64, epochs=15, validation_data=(x_test, y_test))
#     model.save_weights('weights.h5')


def getSport(predict):
    if predict[0] == max(predict):
        return 0
    elif predict[1] == max(predict):
        return 1
    elif predict[2] == max(predict):
        return 2
    elif predict[3] == max(predict):
        return 3
    elif predict[4] == max(predict):
        return 4
    elif predict[5] == max(predict):
        return 5

def writeToExcel(model):
    test = []
    for imgName in os.listdir(testDataPath):
        path = os.path.join(testDataPath, imgName)
        image = imread(path)
        image = np.array(resize(image, (imgSize, imgSize, 3)))
        image = image.reshape(1, imgSize, imgSize, 3)
        predict = model.predict(image)[0]  # prediction return with a list of values first one is the prediction
        test.append([imgName, getSport(predict)])

    test = np.array(test)
    for i in test:
        with open('Submit5.csv', mode='a') as IMG:
            IMG_ = csv.writer(IMG, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            IMG_.writerow(i)

#writeToExcel(model)

while True:
    x = input('Pls, Enter number of image or To Exit Press (-1): ')
    if x == '-1':
        break
    elif os.path.exists(testDataPath + x + '.jpg'):
        img = imread(testDataPath + x + '.jpg')
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.imshow(img)
        plt.show()
        img = np.array(resize(img, (imgSize, imgSize, 3)))
        img = img.reshape(1, imgSize, imgSize, 3)
        prediction = model.predict(img)[0]

        print(f'This Sport is {dic[getSport(prediction)]}.\n')
    else:
        print('This image not exist!!\n')

print('Thanks for your time (^_^)\n')
