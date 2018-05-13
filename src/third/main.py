import numpy as np
import cv2
import cv2.ml as ml
from matplotlib import pyplot as plt


class Application:
    def menu(self):
        print("0 - Read image")
        print('1 - Apply linear filter')
        print("2 - Apply blur(...)")
        print("3 - Apply medianBlur(...)")
        print("4 - Apply GaussianBlur(...)")
        print("5 - Apply erode(...)")
        print("6 - Apply dilate(...)")
        print("7 - Apply Sobel(...)")
        print("8 - Apply Laplacian(...)")
        print("9 - Apply Canny(...)")
        print("10 - Apply calcHist(...)")
        print("11 - Apply equalizeHist(...)")
        print("12 - exit")

        menuIndex = int(input("Write index: "))
        return menuIndex


class StatModel(object):
    '''parent class - starting point to add abstraction'''

    def load(self, fn):
        self.model.load(fn)

    def save(self, fn):
        self.model.save(fn)

    def drawPoints(self, points, responses):
        min, max = self.get_range(points)
        h = 0.02
        xx, yy = np.meshgrid(np.arange(min[0], max[0], h), np.arange(min[1], max[1], h))
        X_hypo = np.c_[xx.ravel().astype(np.float32), yy.ravel().astype(np.float32)]
        zz = self.predict(X_hypo)
        zz = zz.reshape(xx.shape)
        plt.contourf(xx, yy, zz, cmap=plt.cm.coolwarm, alpha=0.8)
        plt.scatter(points[:, 0], points[:, 1], c=np.ravel(responses), s=30)
        plt.show()

    def get_range(self, points):
        max = np.amax(points, axis=0)
        min = np.amin(points, axis=0)
        return (min, max)

    def draw(self, img):
        plt.imshow(img)
        plt.gca().invert_yaxis()
        plt.show()

    def read_file(self, filename):
        fs_read = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
        features_train = fs_read.getNode('features_train').mat()
        response_train = fs_read.getNode('response_train').mat()
        features_test = fs_read.getNode('features_test').mat()
        response_test = fs_read.getNode('response_test').mat()
        fs_read.release()
        return (features_train, response_train, features_test, response_test)

    def predict(self, values):
        result = self.model.predict(values)
        return np.ravel(result[1])

    def error(self, features, responses):
        _responses = np.array(np.ravel(self.predict(features)), dtype=np.int)
        responses = np.ravel(responses)
        return np.sum(responses == _responses) / len(responses)


class SVM(StatModel):
    '''wrapper for OpenCV SimpleVectorMachine algorithm'''

    def __init__(self):
        self.model = ml.SVM_create()

    def train(self, samples, responses):
        # setting algorithm parameters
        self.model.setType(ml.SVM_C_SVC)
        self.model.setKernel(ml.SVM_LINEAR)
        self.model.setC(2)
        self.model.setGamma(2)
        return self.model.train(samples, ml.ROW_SAMPLE, responses)


class DTrees(StatModel):
    '''wrapper for OpenCV SimpleVectorMachine algorithm'''

    def __init__(self):
        self.model = ml.DTrees_create()

    def train(self, samples, responses):
        # setting algorithm parameters
        return self.model.train(samples, ml.ROW_SAMPLE, responses)


class RTrees(StatModel):
    '''wrapper for OpenCV SimpleVectorMachine algorithm'''

    def __init__(self, n_trees=250, min_sample_count=2, max_depth=10):
        self.model = ml.RTrees_create()
        eps = 1
        criteria = (cv2.TERM_CRITERIA_MAX_ITER, n_trees, eps)
        self.model.setTermCriteria(criteria)
        self.model.setMinSampleCount(min_sample_count)
        self.model.setMaxDepth(max_depth)

    def train(self, samples, responses):
        # setting algorithm parameters
        self.model.setMaxCategories(len(np.unique(responses)))
        return self.model.train(samples, ml.ROW_SAMPLE, responses)


class GBTrees(StatModel):
    '''wrapper for OpenCV SimpleVectorMachine algorithm'''

    def __init__(self, n_trees=250, min_sample_count=2, max_depth=10):
        self.model = ml.GbTrees_create()
        eps = 1
        criteria = (cv2.TERM_CRITERIA_MAX_ITER, n_trees, eps)
        self.model.setTermCriteria(criteria)
        self.model.setMaxCategories(len(np.unique(responses)))
        self.model.setMinSampleCount(min_sample_count)
        self.model.setMaxDepth(max_depth)

    def train(self, samples, responses):
        # setting algorithm parameters
        return self.model.train(samples, ml.ROW_SAMPLE, responses)

clf = RTrees()
features, responses, test_features, test_responses = clf.read_file("clusterization/datasetMulticlass.yml")
clf.train(features, responses)
y_val = clf.predict(features)

print(clf.error(test_features, test_responses))
img = clf.drawPoints(features, responses)
