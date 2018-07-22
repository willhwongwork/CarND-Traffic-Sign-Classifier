# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup-images/visualization.png "Visualization"
[image2]: ./writeup-images/grayscale.png "Grayscaling"
[image3]: ./writeup-images/random_noise.jpg "Random Noise"
[image4]: ./traffic-signs-data/traffic-sign.jpg "Traffic Sign 1"
[image5]: ./traffic-signs-data/traffic-sign1.jpg "Traffic Sign 2"
[image6]: ./traffic-signs-data/traffic-sign2.jpg "Traffic Sign 3"
[image7]: ./traffic-signs-data/traffic-sign3.jpg "Traffic Sign 4"
[image8]: ./traffic-signs-data/traffic-sign4.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32*32
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distribute across the 43 labels.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because using grayscale images instead of color improves the ConvNet's accuracy.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because mage data should be normalized so that the data has mean zero and equal variance.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
| Flatten 				| Input = 5x5x16. Output = 400					|
| Fully connected		| Input = 400. Output = 120        				|
| RELU                  |												|
| Dropout               | 0.5											|
| Fully connected		| Input = 120. Output = 84       				|
| RELU                  |												|
| Dropout               | 0.5											|
| Fully connected		| Input = 84. Output = 43        				|
| Softmax				|        										|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used batch size of 128, 20 epochs, learning rate of 0.001. I used the X_train_gray_normalized for training and then compute softmax cross entropy of logits and labels to measure accuracy.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.993
* validation set accuracy of 0.953 
* test set accuracy of 0.926

If a well known architecture was chosen:
* The LeNet architecture was chosen.
* LeNet is a pioneering 7-level convolutional network by LeCun that classifies digits, and both hand-written digits and traffic signs have similar characteristics. 
* The final model's accuracy on the training, validation and test set provide evidence that the model is working well. The training set accuracy is 0.993 meaning the model recognizes almost all of the images it was trained on. The validation set accuracy of 0.953, above 0.93.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first 3 images are relatively easy to classify. The 4th one has undistinguishable colors and shades that might be difficult to classify. The 5th one has more complex shape than the normal number or arrow signs, it might be harder for the classifier to recognize non-geometric sign.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Keep right     		| Keep right  									| 
| Speed limit (70km/h)  | Speed limit (70km/h)							|
| Speed limit (30km/h)	| Speed limit (30km/h)							|
| Turn right ahead      | Turn right ahead 					 			|
| Children crossing		| Children crossing      						|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 92.6%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model is almost 100% sure that this is a Keep right sign (probability of 1.0), and the image does contain a Keep right sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000         		| Keep right   									| 
| .00000     			| Stop 											|
| .00000				| Yield											|
| .00000	      		| Road work					 					|
| .00000				| Go straight or right      					|


For the second image, the model is 99.999% sure that this is a Speed limit (70km/h) sign (probability of 0.99999), and the image does contain a Speed limit (70km/h) sign. The top five soft max probabilities were... 


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99999         		| Speed limit (70km/h)   						| 
| .00001     			| Speed limit (30km/h) 							|
| .00000				| Speed limit (20km/h)							|
| .00000	      		| Speed limit (50km/h)					 		|
| .00000				| Speed limit (80km/h)      					|

For the third image, the model is 100% sure that this is a Speed limit (30km/h) sign (probability of 1.0), and the image does contain a Speed limit (30km/h) sign. The top five soft max probabilities were... 


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000         		| Speed limit (30km/h)   						| 
| .00000     			| Speed limit (20km/h) 							|
| .00000				| Speed limit (50km/h)							|
| .00000	      		| Speed limit (70km/h)					 		|
| .00000				| Speed limit (80km/h)      					|

For the forth image, the model is 99.582% sure that this is a Turn right ahead sign (probability of 0.99582), and the image does contain a Turn right ahead sign. The top five soft max probabilities were... 


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99582         		| Turn right ahead   							| 
| .00381     			| Ahead only 									|
| .00019				| No passing									|
| .00009	      		| Go straight or left					 		|
| .00005				| Yield      									|

For the fifth image, the model is 57.811% sure that this is a Children crossing (probability of 0.57811), and the image does contain a Children crossing sign. The top five soft max probabilities were... 


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .57811         		| Children crossing  							| 
| .28818     			| Bicycles crossing 							|
| .05561				| Dangerous curve to the right					|
| .03744	      		| Beware of ice/snow					 		|
| .02898				| Slippery road      							|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


