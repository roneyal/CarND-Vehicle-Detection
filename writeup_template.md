##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/676.jpg
[image2]: ./examples/984.jpg
[image3]: ./examples/boxes_725.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  



###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I placed all the image processing utility functions in img_process.py. For each image I extracted spatial, color histogram and histogram of gradient features for training and testing.
The function extract_features (line 119) extracts all these features using other utility functions and according to the parameters it receives from the classifier.

Since the features I used have different scales of magnitude, I used a StandardScaler to normalize each column (see lines 98-100 in classifier.py).

I also created a classifier class which handles the preprocessing of images, training an SVC and classifying test images (classifier.py).


I tried to manually select different values for the feature extraction, but I found that even when the classifier performs well on the test set (over 95%) it is not enough for the driving movie.


####2. Explain how you settled on your final choice of HOG parameters.

To improve the classifier I ran a grid search over the parameter values (lines 50 - 81 in classifier.py, commented out). I got the best results on the test set (~99%) with one of the combinations, and from that point I used those values as constants (lines 23-32 in classifier.py). Perhaps most importantly, I worked in the YUV color space. 


####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained the classifier on the vehicle and non-vehicle images which were divided randomly into training and test sets and fed into the SVC (lines 86-115 in classifier.py).
This was a lengthy process so I saved the trained classifier and in subsequent runs I loaded it from disk using "pickle" (see lines 169 - 179 in sliding_window.py).

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I identified representative vehicle sizes in each region in the frame and defines different search areas and window scales (lines 91-152 in sliding_window.py).
I had to add many regions and window sizes to get decent performance in the video, it was basically a manual tuning process where I added and updated the regions after each test run.
Here is an example that shows that the classifier identified the vehicles in more than one scale:


![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Here are some example images where my pipeline performed well:

![alt text][image1]
![alt text][image2]
---

To improve performance, in addition to running grid search on the preprocessing parameters, I tried changing the windows sizes and search areas and adding more windows until I was satisfied with the result.

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

To filter and smooth out the bounding boxes I implemented a moving average over the heat maps, where each frame I added the current heatmap with a weight of 0.3 to the current average with a weight of 0.7 to obtain the new average (lines 23-42 in car_detector.py). To remove false positives, I thresholded the heat map at 0.6 (line 45 in car_detector.py).


### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

