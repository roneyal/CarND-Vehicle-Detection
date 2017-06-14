import time
import img_process
import numpy as np
import glob
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def main():
    clf = classifier()

    cars, notcars = clf.get_training_filenames()

    clf.train(cars, notcars)


class classifier():


    def __init__(self):


        self.color_space = 'YUV' #''HSV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient = 9  # HOG orientations
        self.pix_per_cell = 8  # HOG pixels per cell
        self.cell_per_block = 2  # HOG cells per block
        self.hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
        self.spatial_size = (32, 32)  # Spatial binning dimensions
        self.hist_bins = 16  # Number of histogram bins
        self.spatial_feat = True  # Spatial features on or off
        self.hist_feat = True  # Histogram features on or off
        self.hog_feat = True  # HOG features on or off



    def get_training_filenames(self):
        images = glob.glob('../data/**/*.png', recursive=True)
        cars = []
        notcars = []
        for image in images:
            if 'non-vehicles' in image:
                notcars.append(image)
            else:
                cars.append(image)
        return cars, notcars

    def train(self, cars, notcars):


        # best = 0
        # for color_space in ['LUV', 'HLS', 'YUV', 'YCrCb', 'RGB', 'HSV']:
        #     for spatial_size in [(16,16), (32,32)]:
        #         for hist_bins in [16, 32]:
        #             for orient in [9]:
        #                 for pix_per_cell in [8]:
        #                     for cell_per_block in [2,4]:
        #                         for hog_ch in ['ALL', 0, 1, 2]:
        #                             print (color_space, spatial_size,
        #                                                         hist_bins,
        #                                                         orient,
        #                                                         pix_per_cell,
        #                                                         cell_per_block,
        #                                                         hog_ch)
        #                             score = self.train_internal(cars, notcars,
        #                                                         color_space,
        #                                                         spatial_size,
        #                                                         hist_bins=hist_bins,
        #                                                         orient=orient,
        #                                                         pix_per_cell=pix_per_cell,
        #                                                         cell_per_block=cell_per_block,
        #                                                         hog_channel=hog_ch)
        #                             if score > best:
        #                                 best = score
        #                                 self.color_space = color_space
        #                                 self.spatial_size = spatial_size
        #                                 self.hist_bins = hist_bins
        #                                 self.orient = orient
        #                                 self.pix_per_cell = pix_per_cell
        #                                 self.cell_per_block = cell_per_block
        #                                 self.hog_channel = hog_ch
        #                                 print('best score so far:', score)

        #train again with the best
        self.train_internal(cars, notcars, self.color_space, self.spatial_size, self.hist_bins, self.orient, self.pix_per_cell, self.cell_per_block, self.hog_channel, test_size=0.1)

    def train_internal(self, cars, notcars, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0, test_size=0.2):

        car_features = img_process.extract_features(cars, cspace=color_space, spatial_size=spatial_size,
                                        hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
        notcar_features = img_process.extract_features(notcars, cspace=color_space, spatial_size=spatial_size,
                                           hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
        if len(car_features) > 0:
            # Create an array stack of feature vectors
            X = np.vstack((car_features, notcar_features)).astype(np.float64)
            # Fit a per-column scaler
            self.X_scaler = StandardScaler().fit(X)
            # Apply the scaler to X
            scaled_X = self.X_scaler.transform(X)

            # Define the labels vector
            y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

            # Split up data into randomized training and test sets
            rand_state = np.random.randint(0, 100)
            X_train, X_test, y_train, y_test = train_test_split(
                scaled_X, y, test_size=test_size, random_state=rand_state)

            print('Feature vector length:', len(X_train[0]))
            # Use a linear SVC
            self.svc = LinearSVC()
            # Check the training time for the SVC
            t = time.time()
            self.svc.fit(X_train, y_train)
            t2 = time.time()
            print(round(t2 - t, 2), 'Seconds to train SVC...')
            # Check the score of the SVC
            score = self.svc.score(X_test, y_test)
            print('Test Accuracy of SVC = ', round(score, 4))
            # Check the prediction time for a single sample
            t = time.time()
            n_predict = 10
            print('My SVC predicts: ', self.svc.predict(X_test[0:n_predict]))
            print('For these', n_predict, 'labels: ', y_test[0:n_predict])
            t2 = time.time()
            print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')
            return score

    def classify(self, test_img):
        features = img_process.single_img_features(test_img, color_space=self.color_space, spatial_size=self.spatial_size,
                                                   hist_bins=self.hist_bins, orient=self.orient,
                                                   pix_per_cell=self.pix_per_cell, cell_per_block=self.cell_per_block, hog_channel=self.hog_channel,
                                                   spatial_feat=True, hist_feat=True, hog_feat=True)
        # 5) Scale extracted features to be fed to classifier
        test_features = self.X_scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = self.svc.predict(test_features)
        return prediction


if __name__ == '__main__':
    main()