import time
from img_process import extract_features
import numpy as np
import glob
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def main():
    clf = classifier()

    cars, notcars = clf.get_training_filenames()

    clf.train_clf(cars, notcars)


class classifier():


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


    def train_clf(self, cars, notcars):
        ### HOG parameters
        colorspace = 'RGB'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        orient = 9
        pix_per_cell = 8
        cell_per_block = 2
        hog_channel = 0  # Can be 0, 1, 2, or "ALL"
        ###
        car_features = extract_features(cars, cspace='RGB', spatial_size=(32, 32),
                                        hist_bins=32, hist_range=(0, 256))
        notcar_features = extract_features(notcars, cspace='RGB', spatial_size=(32, 32),
                                           hist_bins=32, hist_range=(0, 256))
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
                scaled_X, y, test_size=0.2, random_state=rand_state)

            print('Feature vector length:', len(X_train[0]))
            # Use a linear SVC
            self.svc = LinearSVC()
            # Check the training time for the SVC
            t = time.time()
            self.svc.fit(X_train, y_train)
            t2 = time.time()
            print(round(t2 - t, 2), 'Seconds to train SVC...')
            # Check the score of the SVC
            print('Test Accuracy of SVC = ', round(self.svc.score(X_test, y_test), 4))
            # Check the prediction time for a single sample
            t = time.time()
            n_predict = 10
            print('My SVC predicts: ', self.svc.predict(X_test[0:n_predict]))
            print('For these', n_predict, 'labels: ', y_test[0:n_predict])
            t2 = time.time()
            print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')



if __name__ == '__main__':
    main()