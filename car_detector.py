
from moviepy.editor import VideoFileClip
import pickle
import numpy as np
import sliding_window
import heatmap
from scipy.ndimage.measurements import label
import scipy #for saving images
import collections

class car_detector():

    def __init__(self):
        self.heat = None
        self.count = 0
        #self.heatmaps = collections.deque(maxlen=10)

    def handle_image(self, img):
        # if self.count < 170:
        #     self.count += 1
        #     return img

        alpha = 0.3
        #scipy.misc.imsave('input_images/' + str(self.count) + '.jpg', img)
        if self.heat is None:
            self.heat = np.zeros_like(img[:, :, 0]).astype(np.float)
        else:
            self.heat = self.heat * (1.0 - alpha)

        heat = np.zeros_like(img[:, :, 0]).astype(np.float)

        image = img.astype(np.float32) / 255.
        boxes = sliding_window.search_classify(image, clf)
        draw_image = np.copy(image)

        window_img = sliding_window.draw_boxes(draw_image, boxes, color=(0, 0, 1.0), thick=6)
        scipy.misc.imsave('output_images/boxes_' + str(self.count) + '.jpg', window_img)

        heat = heatmap.add_heat(heat, boxes)
        #self.heatmaps.append(heat)

        self.heat += heat * alpha

        # Apply threshold to help remove false positives
        hm = heatmap.apply_threshold(np.copy(self.heat), 0.6)

        # Visualize the heatmap when displaying
        hm = np.clip(hm, 0, 255)

        scipy.misc.imsave('output_images/map_' + str(self.count) + '.jpg', self.heat)
        # Find final boxes from heatmap using label function
        labels = label(hm)
        draw_img = heatmap.draw_labeled_bboxes(image, labels)

        scipy.misc.imsave('output_images/' + str(self.count) + '.jpg', draw_img)
        self.count += 1

        return draw_img * 255.0


def run_video():

    clip1 = VideoFileClip("project_video.mp4")
    detector = car_detector()
    out_clip = clip1.fl_image(detector.handle_image)
    out_clip.write_videofile('out.mp4', audio=False)



if __name__ == '__main__':

    with open("classifier.pkl", "rb") as file:
        clf = pickle.load(file)
        print('loaded clf')
    run_video()