#**Behavioral Cloning** 

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

[image1]: ./examples/placeholder.png "Model Visualization"
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

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

Before implementing CNN for my model I have pre-processed data by cropping and normalizing images. As the upper part contains trees and hills which adds no value to training the car, 50px on top is cropped and 20px on bottom to remove hood of the car is cropped. This makes the input to convolutional layer 90x320x3.

I have also normalized the data so that the value of a pixel will be in between -0.5 to +0.5.

My model consists of a convolution neural network with following layers:

| Layer         										 |
|:------------------------------------------------------:|
| Input 90x320x3 - 3 channel image  					 |
| Convolution 5x5, 2x2 stride, Depth 24, RELU activation | 
| Convolution 5x5, 2x2 stride, Depth 36, RELU activation |
| Convolution 5x5, 2x2 stride, Depth 48, RELU activation |
| Convolution 3x3, 2x2 stride, Depth 64, RELU activation |
| Convolution 3x3, 2x2 stride, Depth 64, RELU activation |
| Flatter												 |
| Fully connected 	Output = 100			         	 |
| Fully connected	Output = 50	        				 |
| Fully connected	Output = 1	        				 |

Stride in Tensorflow = subsample in keras

#### 2. Attempts to reduce overfitting in the model

I have shuffled the training data to reduce overfitting before starting batch processing and also at the time of yielding the output from generator.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

There is one steering angle correction parameter for left and right side images which is tuned as 2.0.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of centre lane driving, recovering from the left and right sides of the road. The model will treat all three kind of images as if they are coming from centre camera.

So suppose the car sees one image coming from centre camera similar to one we get with left camera then it will add 2.0 in steering angle to move it on right side a bit and it will subtract 2.0 when it will see image similar to one coming from right camera.

This way we will be able to keep the car on centre of the road when it will drift to right or left.

Another way to keep the car in centre is to collect data in a way that it is recovering from side of the road to centre. This can be achieved nicely by turning the recorder ON only when it is recovering to centre from sides and not while car is drifting away from centre. This way it will learn not learn to drift away from road and will be on track properly.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to train the car to run smoothly in autonomous mode.

My first step was to use a convolution neural network model. I used nvidia model of deep learning because it is very powerful and it contains 9 layers as shown in table 1.

I used 2 Epochs initially to see how well it works. But in second Epochs my validation loss got increased instead of reducing. Then, I kept only 1 Epoch and it worked well.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
