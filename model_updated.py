import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import sklearn

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Cropping2D, BatchNormalization
from keras.layers.core import Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, Callback


def preprocess_image(img):
    '''
    Preprocessing images: same used in drive.py. However convert from BGR to YUV
    '''
    # original shape: 160x320x3 ==> Objective shape for neural net: 66x200x3
    # crop to 90x320x3
    new_image = img[50:140,:,:]
    # applying noise
    new_image = cv2.GaussianBlur(new_image, (3,3), 0)
    # resize to new scale to 66x200x3 (same as nVidia)
    new_image = cv2.resize(new_image,(200, 66), interpolation = cv2.INTER_AREA)
    # convert to YUV color space (as nVidia paper suggests)
    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2YUV)
    return new_image

def random_distort(img, measurement):
    '''
    method for adding random distortion, random brightness, and random vertical
    shift  to dataset images.
    '''
    new_img = img.astype(float)
    # random brightness - the mask bit keeps values from going beyond (0,255)
    value = np.random.randint(-28, 28)
    if value > 0:
        mask = (new_img[:,:,0] + value) > 255
    if value <= 0:
        mask = (new_img[:,:,0] + value) < 0
    new_img[:,:,0] += np.where(mask, 0, value)
    # random shadow - full height, random left/right side, random darkening
    h,w = new_img.shape[0:2]
    mid = np.random.randint(0,w)
    factor = np.random.uniform(0.55,0.75)
    if np.random.rand() > .5:
        new_img[:,0:mid,0] *= factor
    else:
        new_img[:,mid:w,0] *= factor
    # randomly shift horizon
    h,w,_ = new_img.shape
    horizon = 2*h/5
    v_shift = np.random.randint(-h/8,h/8)
    pts1 = np.float32([[0,horizon],[w,horizon],[0,h],[w,h]])
    pts2 = np.float32([[0,horizon+v_shift],[w,horizon+v_shift],[0,h],[w,h]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    new_img = cv2.warpPerspective(new_img,M,(w,h), borderMode=cv2.BORDER_REPLICATE)
    return (new_img.astype(np.uint8), measurement)


def generate_training_data(images_path, measurements, batch_size=128):
    '''
    method for the model training data generator performing the next tasks:
    -> load,
    -> process, and
    -> distort images,
    -> using yield  data to model
    In some cases some images are flipped
    '''
    images_path, measurements = shuffle(images_path, measurements)
    X,y = ([],[])
    while True:
        for i in range(len(measurements)):
            #print(images_path[i])
            img = cv2.imread(images_path[i])
            measurement = float(measurements[i])
            img = preprocess_image(img) # pre-processing Image
            img, measurement = random_distort(img, measurement) # Random Distortion of image
            X.append(img)
            y.append(measurement)
            if len(X) == batch_size:
                yield sklearn.utils.shuffle(np.array(X), np.array(y))
                X, y = ([],[])
                images_path, measurements = shuffle(images_path, measurements)
            # flip horizontally and invert steer angle, if magnitude is > 0.33
            if abs(measurement) > 0.35:
                img = cv2.flip(img, 1)
                measurement *= -1
                X.append(img)
                y.append(measurement)
                if len(X) == batch_size:
                    yield sklearn.utils.shuffle(np.array(X), np.array(y))
                    X, y = ([],[])
                    images_path, measurements = shuffle(images_path, measurements)

# ############################################################################
# Acessing to Udacity and/or custom Data
# Note:
# -> Path's information should be added in vector 'csv_files_vector'
# -> images of new training data should be copied to ../IMG/ folder
# ###########################################################################

#csv_files_vector = ['../data/driving_log.csv','../custome_data/driving_log.csv','../custome_data_mountains/driving_log.csv','../custome_data_mountains1/driving_log.csv']
#csv_files_vector = ['../custome_data_mountains/driving_log.csv','../custome_data_mountains1/driving_log.csv']
#csv_files_vector = ['../custome_data_mountains1/driving_log.csv']
#csv_files_vector = ['../data/driving_log.csv']
csv_files_vector = ['../data/driving_log.csv','../custome_data/driving_log.csv']

# Flag to plot and  understand data training distribution
flag_plot_input_data = False

############################################################################
# Getting data that will be used in training, validation
###########################################################################

lines = []
for jj in range(len(csv_files_vector)):
    print(jj)
    print(csv_files_vector[jj])
    with open(csv_files_vector[jj]) as csvfile:
        reader = csv.reader(csvfile)
        next(reader) #to Skip the first line that contains header info
        for line in reader:
            lines.append(line)


images_path = []
measurements = []

for line in lines:
    # Getting information of 3 cameras ( center, left, right)
    for i in range(3):
        # For training scenarios that in some cases speed < 0.1
        # in order to avoid learning scenrios in which the car is stopped
        if float(line[6]) < 0.1:
            continue
        source_path =line[i]
        filename = source_path.split('/')[-1]
        current_path = '../data/IMG/' + filename
        #image = cv2.imread(current_path)
        images_path.append(current_path)
        if (i==0): # Center Camera
            offset = 0
        elif (i==1): # Left Camera
            offset = 0.2
        else: # Right Camera
            offset = -0.2
        measurement = float(line[3])
        measurements.append(measurement+offset)

images_path = np.array(images_path)
measurements = np.array(measurements)

print('')
print('Before Data Distribution pre Processing: ', images_path.shape, measurements.shape)
total_samples = len(images_path)
print(total_samples)

# print a histogram to check measurement representation
number_of_bins = 23
average_sampls_per_bin = len(measurements)/number_of_bins
hist, bins = np.histogram(measurements, number_of_bins)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2

if flag_plot_input_data:
    plt.bar(center, hist, align='center', width=width)
    plt.plot((np.min(measurements), np.max(measurements)), (average_sampls_per_bin, average_sampls_per_bin), 'k-')
    plt.title('Dataset distribution before data filter')
    plt.show()

# Reducing the number of bins that have ranges overreprsented
probabilities = []
target = average_sampls_per_bin * .5
for i in range(number_of_bins):
    if hist[i] < target:
        probabilities.append(1.)
    else:
        probabilities.append(1./(hist[i]/target))
remove_list = []
for i in range(len(measurements)):
    for j in range(number_of_bins):
        if measurements[i] > bins[j] and measurements[i] <= bins[j+1]:
            if np.random.rand() > probabilities[j]:
                remove_list.append(i)
images_path = np.delete(images_path, remove_list, axis=0)
measurements = np.delete(measurements, remove_list)

# showing new measurement distribution (objective: get equal representation for each stearing value)
hist, bins = np.histogram(measurements, number_of_bins)
if flag_plot_input_data:
    plt.bar(center, hist, align='center', width=width)
    plt.plot((np.min(measurements), np.max(measurements)), (average_sampls_per_bin, average_sampls_per_bin), 'k-')
    plt.title('Dataset distribution after data filter')
    plt.show()

print('After Equalization of Data Distribution:', images_path.shape, measurements.shape)

# Dividing data in two groups, trainig group and test group.
images_path_train, images_path_test, measurements_train, measurements_test = train_test_split(images_path, measurements,test_size=0.05, random_state=42)
print('')
print('Train size data:', images_path_train.shape, measurements_train.shape)
print('Test size data:', images_path_test.shape, measurements_test.shape)


################################################################################
# Neural Network Architecture
###############################################################################
model = Sequential()
# input normalized and centered using Nvidia suggested dimensions 66 x 200 x 3

model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(66,200,3)))

dropout_factor = 0.5
# 5x5 convolutional Neural Networks with 24, 36, 48, 64 and 64 layers each one
# and stride of 2x2 and Acvitaion Function 'relu'
#model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
#model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
#model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
#model.add(Convolution2D(64,3,3, activation='relu'))
#model.add(Convolution2D(64,3,3, activation='relu'))
model.add(BatchNormalization(mode=2, axis=1))

model.add(Convolution2D(24,5,5,W_regularizer=l2(0.01)))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(dropout_factor))
model.add(Activation('relu'))

model.add(Convolution2D(36,5,5))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(dropout_factor))
model.add(Activation('relu'))

model.add(Convolution2D(48,5,5))
#model.add(MaxPooling2D((2,2)))
model.add(Dropout(dropout_factor))
model.add(Activation('relu'))

model.add(Convolution2D(48,5,5))
model.add(Dropout(dropout_factor))
model.add(Activation('relu'))

model.add(Convolution2D(48,5,5))
model.add(Activation('relu'))

# Full Neural Networks
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50,W_regularizer=l2(0.01)))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))

################################################################################
# Compilation of Neural Networks
################################################################################
model.compile(loss='mse', optimizer='adam')

# Initialize Generators using yjeld
train_sample = generate_training_data(images_path_train, measurements_train, batch_size=64)
val_sample = generate_training_data(images_path_train, measurements_train, batch_size=64)
test_sample = generate_training_data(images_path_test, measurements_test, batch_size=64)

checkpoint = ModelCheckpoint('model{epoch:02d}.h5')

history = model.fit_generator(train_sample, validation_data=val_sample, nb_val_samples=int(math.ceil(total_samples/10)), samples_per_epoch=total_samples, nb_epoch=20, verbose=1, callbacks=[checkpoint])
#history = model.fit_generator(train_sample, validation_data=val_sample, nb_val_samples=2400, samples_per_epoch=24000, nb_epoch=30, verbose=1, callbacks=[checkpoint])
print('Test Loss result (using data that was not used during training):', model.evaluate_generator(test_sample, 128))
print(model.summary())

model.save('model.h5')
