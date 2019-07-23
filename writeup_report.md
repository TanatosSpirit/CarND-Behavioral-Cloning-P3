# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/center.jpg "Center camera"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline 
I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 24 and 64 (model.py lines 54-65) 

The model includes RELU layers to introduce nonlinearity (code line 56-60), and the data is normalized in the model using a Keras lambda layer (code line 55). 

#### 2. Attempts to reduce overfitting in the model

The model doesn't contains dropout layers. 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 67).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a center lane driving. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to checking 
existing architectures.

My first step was to use a convolution neural network model similar to 
the model described in [Nvidia paper](https://arxiv.org/pdf/1604.07316.pdf). 
I thought this model might be appropriate because the model was successfully 
used by the Nvidia team.

In order to gauge how well the model was working, 
I split my image and steering angle data into a training and 
validation set. I found that my first model had a low mean squared 
error on the training set but a high mean squared error on the validation 
set. This implied that the model was overfitting. 

Despite this, the car could drive several meters in a straight line.

I looked at the dataset and realized that the hood of the car falls into the 
input of the neural network. Therefore, I increased the parameters of 
Cropping2D (model.py line 54).

After that, the car could drive on straight and smooth turn but it left in 
a turn with a small radius outside.

Since the conversion parameter for side cameras could not be accurate and 
the car went outside, I decided to change the correction parameter (line 22).

It turned out that it was necessary to increase the parameter.

At the end of the process, the vehicle is able to drive autonomously 
around the track without leaving the road. [Track recording](https://www.youtube.com/watch?v=tTtD4a04DkI)

#### 2. Final Model Architecture

The final model architecture (model.py lines 54-65) consisted of a 
convolution neural network with the following layers and layer sizes:


Input image of size 160x320x3
1. Layer 1 : Cropping layer to remove uninteresting information
2. Layer 2 : Lambda layer with a lambda function to normalize data
3. Layer 3 : Convolutional layer using 24 of size 5x5 filters followed by RELU activation
4. Layer 4 : Convolutional layer using 36 of size 5x5 filters followed by RELU activation
5. Layer 5 : Convolutional layer using 48 of size 5x5 filters followed by RELU activation
6. Layer 6 : Convolutional layer using 64 of size 3x3 filters followed by RELU activation
7. Layer 7 : Convolutional layer using 64 of size 3x3 filters followed by RELU activation
8. Layer 8 : Flatten layer
9. Layer 9: Fully connected layer with 2112 neurons
10. Layer 10 : Fully connected layer with 100 neurons
11. Layer 11 : Fully connected layer with 50 neurons
12. Layer 12 : Fully connected layer with 10 neurons
13. Layer 13 : Fully connected layer with 1 neurons

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I recorded two laps on track one 
using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

After the collection process, I had 7371 number of data points. 
I then added preprocessed layers to cropping and normalization data.

I finally randomly shuffled the data set and put 20% of the data 
into a validation set. 

I used this training data for training the model. The validation 
set helped determine if the model was over or under fitting. The 
ideal number of epochs was 20 as evidenced by [video result](https://www.youtube.com/watch?v=tTtD4a04DkI) 
and run1.mp4 in repository. I used an adam optimizer 
so that manually training the learning rate wasn't necessary.
