
[image1]: ./data_pictures/cnn_architecture.png "NN_architecture"
[image2]: ./data_pictures/data_set_BEFORE.png "DataSetBeforeProcessing"
[image3]: ./data_pictures/data_set_AFTER.png "DataSetAfterProcessing"
[image4]: ./data_pictures/nVidia_model.png "nVidia architecture"



# Behavioral-clonning Project
A Neural network (CNN and NN) for end-to-end driving in virtual track, using  Keras and TensorFlow

In order to clone driving behavior most of the deep learning knowledge provided in Nano Degree of Self Driving Car Engineer was applied using Keras and TensorFlow APIs. This submission project is composed of 5 main deliveries, listed in following lines.

* File: model.py
* File: model.h5
* File: drive.py
* File: video.mp4
* This writeup

This writeup report is composed of three sections, the first section describes the main code functions, the second section describes the architecture and training strategy, and the last section describes simulation results

## Code Description
the code of this project is based on typical Python libraries as os, csv, cv2, numpy, matplotlib, math, sklearn. It was also used keras libraries, for example, keras.models, keras.layers, and keras.callbacks in order to define important Neural Network functions as Convolution2D, MaxPooling2D, Dense and others.

The main segment of the code begins in the line 100 with a brief description of the dataset definitions. It is possible to use many source datasets obtained of many data obtained with the driving car simulator in training mode. All image information are saved in the ../IMG/ folder in spite of the scenario, it is because csv files contain paths information for every scenario.

In this case we used the vector 'csv_files_vector' with the next paths:
* '../data/driving_log.csv' that contains the dataset provided by Udacity team
* '../custome_data/driving_log.csv' that contains custom dataset generated in the first track considering diverse scenarios
* '../custome_data_mountains/driving_log.csv' that contains custom dataset generated in the second track
* '../custome_data_mountains1/driving_log.csv' that contains custom dataset generated in the second track

The next step is to get dataset information in numpy arrays considering the suggested offset for every angle (called measurement in the code) information for left and right cameras. It is also applied a simple filter in the dataset in order to avoid data with speed < 0.1. An additional action that was performed is to equalize the samples quantity of every angle used to train the model.

![alt text][image2]

![alt text][image3]

The next section of the code is focused to define the Neural Network architecture, in this case in order to facilitate the learning process, we used the Nvidia model suggested by this online course instructors.

Before to train the neural network model, we used a function called 'generate_training_data' that generate variables using the function yield instead of return (it was done following the suggestions of Udacity team).

Finally, the model is saved in the file model.h5

## Architecture and Training Strategy
In the previous section we performed a general description of the code, in this section is done a brief description of the Neural Network architecture used in this project.

Dataset was split in order to generate 3 groups for training, validation, and test. The architecture used Adam optimizer. The training result was done successfully considering 5 epochs. As our training result does not show any over-fitting (compatible results between training and evaluation losses), it was not necessary to use any dropout layer

The input dataset of the architecture was normalized. The following layers (suggested by the NVIDIA model) define the architecture described in detail in Fig. 1

.![alt text][image1]

Our main reference was the NVIDIA architecture that is shown in the next Figure

.![alt text][image4]

It is important to mention that the input dataset were cropped and resized to 66x200 as suggested by NVIDIA model.

During the training process it was evident the high performance of the learning process of the neural network when camera images were resized from 160x320x3  to  66x200x3. Other important issue is that a Gaussian noise were added to dataset used for training.

## Simulation Results
After training the Python code generated the model file called 'model.h5' that was used to run the driving simulator to control the car in autonomous mode. The result could be visualized in the next video. In all cases the car is always in the middle of the road without any incident that hypothetical could injure the car human user.

[![run1](https://img.youtube.com/vi/SlR43dn2uf0/0.jpg)](https://www.youtube.com/watch?v=SlR43dn2uf0 "Track_1")


As a complement, we used the same training code to create a model for self-driving car simulator in the second track. The result were satisfactory, in this particular case we increased the dataset used to train the model and considered 25 epochs. It is possible to check the result in the next video.

[![run2](https://img.youtube.com/vi/XGjJAwd4Phc/0.jpg)](https://www.youtube.com/watch?v=XGjJAwd4Phc "Track_2")





