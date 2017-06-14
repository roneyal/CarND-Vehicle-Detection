import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import img_process
import classifier

# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf):
    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64,64))

        #plt.imshow(img)
        #plt.imshow(test_img)
        prediction = clf.classify(test_img)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows


def search_classify(image, clf):


    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    # image = image.astype(np.float32)/255




    windows = slide_window(image, x_start_stop=[400, None], y_start_stop=[400, 600],
                           xy_window=(50, 50), xy_overlap=(0.1, 0.1))

    hot_windows = search_windows(image, windows, clf)

    windows = slide_window(image, x_start_stop=[400, None], y_start_stop=[400, 600],
                           xy_window=(60, 60), xy_overlap=(0.1, 0.1))

    hot_windows += (search_windows(image, windows, clf))

    windows = slide_window(image, x_start_stop=[400, None], y_start_stop=[400, 600],
                           xy_window=(75, 75), xy_overlap=(0.1, 0.1))

    hot_windows += (search_windows(image, windows, clf))

    windows = slide_window(image, x_start_stop=[400, None], y_start_stop=[400, 600],
                           xy_window=(85, 85), xy_overlap=(0.1, 0.1))

    hot_windows += (search_windows(image, windows, clf))
    windows = slide_window(image, x_start_stop=[400, None], y_start_stop=[400, 600],
                           xy_window=(100, 100), xy_overlap=(0.1, 0.1))

    hot_windows += (search_windows(image, windows, clf))

    windows = slide_window(image, x_start_stop=[400, None], y_start_stop=[400, 620],
                           xy_window=(110, 110), xy_overlap=(0.1, 0.1))

    hot_windows += (search_windows(image, windows, clf))

    windows = slide_window(image, x_start_stop=[400, None], y_start_stop=[400, 620],
                           xy_window=(125, 125), xy_overlap=(0.1, 0.1))

    hot_windows += (search_windows(image, windows, clf))

    windows = slide_window(image, x_start_stop=[400, None], y_start_stop=[380, 680],
                           xy_window=(150, 150), xy_overlap=(0.1, 0.1))

    hot_windows += (search_windows(image, windows, clf))

    windows = slide_window(image, x_start_stop=[400, None], y_start_stop=[380, 680],
                           xy_window=(180, 180), xy_overlap=(0.1, 0.1))

    hot_windows += (search_windows(image, windows, clf))


    windows = slide_window(image, x_start_stop=[400, None], y_start_stop=[350, 680],
                           xy_window=(200, 160), xy_overlap=(0.1, 0.1))

    hot_windows += (search_windows(image, windows, clf))

    windows = slide_window(image, x_start_stop=[400, None], y_start_stop=[350, 680],
                           xy_window=(220, 180), xy_overlap=(0.1, 0.1))

    hot_windows += (search_windows(image, windows, clf))

    windows = slide_window(image, x_start_stop=[400, None], y_start_stop=[350, 680],
                           xy_window=(250, 200), xy_overlap=(0.1, 0.1))

    hot_windows += (search_windows(image, windows, clf))

    windows = slide_window(image, x_start_stop=[400, None], y_start_stop=[350, 680],
                           xy_window=(300, 230), xy_overlap=(0.1, 0.1))

    hot_windows += (search_windows(image, windows, clf))

    return  hot_windows

import pickle

if __name__ == '__main__':
    # image = mpimg.imread('test_images/test1.jpg')
    # windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[None, None],
    #                        xy_window=(128, 128), xy_overlap=(0.5, 0.5))
    #
    # window_img = draw_boxes(image, windows, color=(0, 0, 255), thick=6)
    # plt.imshow(window_img)


    # clf = classifier.classifier()
    # cars, notcars = clf.get_training_filenames()

    # clf.train(cars, notcars)
    # with open( "classifier.pkl", "wb" ) as file:
    #     s = pickle.dump(clf, file)
    #     print(s)

    with open("classifier.pkl", "rb") as file:
        clf = pickle.load(file)
        print('loaded clf')

    for imgfile in ['input_images/258.jpg']:
        image = mpimg.imread(imgfile)
        image = image.astype(np.float32) / 255.
        boxes = search_classify(image, clf)
        draw_image = np.copy(image)
        window_img = draw_boxes(draw_image, boxes, color=(0, 0, 255), thick=6)
        plt.imshow(window_img)
        plt.show()
