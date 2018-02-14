# Behavioral-clonning Project
A Neural network (CNN and NN) for end-to-end driving in virtual track, using  Keras and TensorFlow

In order to clone driving behaviour most of the deep learning knowledge provided in Nano Degree of Self Driving Car Engineer was applied using Keras and TensorFlow APIs. This submission project is composed of 5 main deliveries, listed in following lines.

* File: model.py
* File: model.h5
* File: drive.py
* File: video.mp4
* This writeup

This writeup report is composed of three sections, the first section describes the main code functions, the second section describes the architecture and training strategy, and the last section describes simulation results

## Code Description
the code of this project is based on typical Python libraries as os, csv, cv2, numpy, matplotlib, math, sklearn. It was also used keras libraries, for example, keras.models, keras.layers, and keras.callbacks in order to define important Neural Network functions as Convolution2D, MaxPooling2D, Dense and others.

The main segment of the code begings in the line 100 with a brief description of the dataset definitions. It is possible to use many source datasets obtained of many data obtained with the driving car simulator in training mode. All image information are saved in the ../IMG/ folder in spite of the scenario, it is because csv files contai paths information of every scenario.

In this case we used the vector 'csv_files_vector' with the next paths:
* '../data/driving_log.csv' that contains the dataset provided by udacity team
* '../custome_data/driving_log.csv' that contains custom dataset generated in the first track considering diverse scenarios
* '../custome_data_mountains/driving_log.csv' that contains custom dataset generated in the second track
* '../custome_data_mountains1/driving_log.csv' that contains custom dataset generated in the second track

The next step is to get dataset information in numpy arrays considering the suggested offset for every angle (called measurement in the code) information for left and right cameras. It is also applied a simple filter in the dataset in order to avoid data with speed < 0.1. An aditional action that was peformed is to equalize the samples quantity of every angle used to train the model.

The next setion of the code is focused to define the Neural Network architecture, in this case in order to facilitate the learning process, we used the Nvidia model suggested by this course.

Before to train the neural network model, we used a function called 'generate_training_data' that generate variables using the function yield instead of return (it was done following the suggestions of Udacity team).

Finally, the model is saved in the file model.h5

## Architecture and Training Strategy
In the previous section we performed a general description of the code, in this section is done a brief description of the Neural Network architecture used in this projet.

In the input of the architecture is performed a normalization of the dataset. Following layers are defined by the NVIDIA architecture with the shape defined in the Fig. 1



## Simulation Results
