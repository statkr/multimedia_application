import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt

# img = cv2.imread('logo.png',0)
# # Initiate ORB detector
# orb = cv2.ORB_create()
# # find the keypoints with ORB
# kp = orb.detect(img,None)
# # compute the descriptors with ORB
# kp, des = orb.compute(img, kp)
# # draw only keypoints location,not size and orientation
# img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
# plt.imshow(img2), plt.show()

from os import listdir
from os.path import isfile, join


class Application:
    def __init__(self, extractor, detector):
        self.extractor = extractor
        self.detector = detector

    def train_vocabulary(self, file_list, vocabulary_size):
        kmeans_trainer = cv2.BOWKMeansTrainer(vocabulary_size)
        for path_to_image in file_list:
            img = cv2.imread(path_to_image, 0)
            kp, des = self.detector.detectAndCompute(img, None)
            kmeans_trainer.add(des)
        return kmeans_trainer.cluster()

    def extract_features_from_image(self, file_name):
        image = cv2.imread(file_name)
        return self.extractor.compute(image, self.detector.detect(image))

    def extract_train_data(self, file_list, category):
        train_data, train_responses = [], []
        for path_to_file in file_list:
            train_data.extend(self.extract_features_from_image(path_to_file))
            train_responses.append(category)
        return train_data, train_responses

    def train_classifier(self, data, responses):
        n_trees = 200
        min_sample_count = 2
        max_depth = 10
        model = cv2.ml.RTrees_create()
        eps = 1
        criteria = (cv2.TERM_CRITERIA_MAX_ITER, n_trees, eps)
        model.setTermCriteria(criteria)
        model.setMaxDepth(max_depth)
        model.train(np.array(data), cv2.ml.ROW_SAMPLE, np.array(responses))
        return model

    def predict(self, file_name):
        features = self.extract_features_from_image(file_name)
        return self.classifier.predict(features)[0]

    def train(self, files_array, vocabulary_size=12):
        all_categories = []
        for category in files_array:
            all_categories += category

        vocabulary = self.train_vocabulary(all_categories, vocabulary_size)
        self.extractor.setVocabulary(vocabulary)

        data = []
        responses = []
        for id in range(len(files_array)):
            data_temp, responses_temp = self.extract_train_data(files_array[id], id)
            data += data_temp
            responses += responses_temp

        self.classifier = self.train_classifier(data, responses)

    def error(self, file_list, category):
        responses = np.array([self.predict(file) for file in file_list])
        _responses = np.array([category for _ in range(len(responses))])
        return 1 - np.sum(responses == _responses) / len(responses)


def get_images_from_folder(folder):
    return ["%s/%s" % (folder, f) for f in listdir(folder) if isfile(join(folder, f))]


def start(folders, detector_type, voc_size, train_proportion):
    if detector_type == "SIFT":
        extract = cv2.xfeatures2d.SIFT_create()
        detector = cv2.xfeatures2d.SIFT_create()
    else:
        extract = cv2.xfeatures2d.SURF_create()
        detector = cv2.xfeatures2d.SURF_create()
    flann_params = dict(algorithm=1, trees=5)
    matcher = cv2.FlannBasedMatcher(flann_params, {})
    extractor = cv2.BOWImgDescriptorExtractor(extract, matcher)

    train = []
    test = []
    for folder in folders:
        images = get_images_from_folder(folder)
        np.random.shuffle(images)

        slice = int(len(images) * train_proportion)
        train_images = images[0:slice]
        test_images = images[slice:]
        train.append(train_images)
        test.append(test_images)

    app = Application(extractor, detector)
    app.train(train, voc_size)

    for id in range(len(test)):
        print(app.error(test[id], id))


firstFolder = sys.argv[1]
secondFolder = sys.argv[2]
detectorType = sys.argv[3]
vocSize = int(sys.argv[4])
trainProportion = float(sys.argv[5])

start([firstFolder, secondFolder], detectorType, vocSize, trainProportion)
