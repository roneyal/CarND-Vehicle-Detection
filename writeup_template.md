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
[image4]: ./examples/60.jpg
[image5]: ./examples/map_60.jpg
[image6]: ./examples/194.jpg
[image7]: ./examples/map_194.jpg
[image8]: ./examples/boxes_194.jpg
[video1]: ./project_video.mp4

###Here is my writeup for this project!

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

For example, here is a frame with the corresponding averaged heat map. Thanks to the thresholding, there is no bounding box drawn on the frame:

![alt text][image4]
![alt text][image5]

Here is another example of the detections in a single frame:

![alt text][image8]

Here is the averaged heat map for that frame:

![alt text][image7]

And this is the final outcome after combining and thresholding:

![alt text][image6]


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I think the main problem I saw in this exercise it that the training data is very different from the test data (video). So despite the fact that I divided the training data into a training set and test set, the performance out of the box on the video was not good. Unfortunately, I didn't have time to train on more data. In addition, data augmentation in the training phase. might have helped the classifier generalize.

I had to tweak many parameters, such as the window sizes and search areas and the heat-map threshold to match the given video, but I'm sure that a video in another setting or different lighting conditions would expose other failure points of the model.

To make it more robust, perhaps I could use in the future a combination of algorithms. I believe that combining deep learning with the classical approach that we used in this exercise could yield good results and I hope to test this out in the future.



