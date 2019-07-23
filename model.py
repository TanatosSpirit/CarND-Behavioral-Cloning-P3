import csv
import cv2
import numpy as np

# Loading dataset
lines = []
header = True

path = "E://datasets//udacity-behavior-cloning//driving-data//"

with open(path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if header:
            header = False
            continue
        lines.append(line)

images = []
measurements = []

correction = 0.3  # this is a parameter to tune

for line in lines:
    center_image_filename = line[0]
    left_image_filename = line[1]
    right_image_filename = line[2]

    steering_center = float(line[3])

    # create adjusted steering measurements for the side camera images
    steering_left = steering_center + correction
    steering_right = steering_center - correction

    # read in images from center, left and right cameras
    img_center = cv2.imread(center_image_filename)
    img_left = cv2.imread(left_image_filename)
    img_right = cv2.imread(right_image_filename)

    images.extend([img_center, img_left, img_right])
    measurements.extend([steering_center, steering_left, steering_right])

X_train = np.array(images)
y_train = np.array(measurements)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from  keras.layers.convolutional import Convolution2D

#create model NvidiaNet
model = Sequential()

model.add(Cropping2D(cropping=((66, 26), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Convolution2D(filters=24, kernel_size=5, strides=(2, 2), activation='relu'))
model.add(Convolution2D(filters=36, kernel_size=5, strides=(2, 2), activation='relu'))
model.add(Convolution2D(filters=48, kernel_size=5, strides=(2, 2), activation='relu'))
model.add(Convolution2D(filters=64, kernel_size=3, activation='relu'))
model.add(Convolution2D(filters=64, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

#Training
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=20)

model.summary()

model.save('model.h5')
