import numpy as np
import cv2
import cv2.ml as ml
from matplotlib import pyplot as plt


class Application:
    def __init__(self):
        self.features = None
        self.responses = None
        self.test_features = None
        self.test_responses = None

    def menu(self):
        print("Menu")
        print("0 - Read file")
        print("1 - Choose algorithm")

        menuIndex = int(input("Write index: "))
        return menuIndex

    def start(self):

        while True:
            index = self.menu()
            if (index == 0):
                filename = input("Write filename: ")
                self.read_file("clusterization/" + filename)
            if (index == 1):
                print("Choose algorithm")
                print("1 - SVM")
                print("2 - RTree")
                print("3 - DTree")
                alg = int(input("Number: "))
                if (alg == 1):
                    clf = self.createSVM()
                elif alg == 3:
                    clf = self.createDtree()
                else:
                    clf = self.createRtree()
                clf.train(self.features_train, self.response_train)
                print("Error train: %f" % clf.error(self.features_train, self.response_train))
                print("Error test: %f" % clf.error(self.features_test, self.response_test))
                try:
                    clf.drawPoints(self.features_train, self.response_train)
                except:
                    pass

    def createSVM(self):
        print("Choose type:")
        print("0 - C_SVC")

        print("1 - NU_SVC")
        print("2 - ONE_CLASS")
        print("3 - EPS_SVR")
        index = int(input("Write index [0]: "))
        type = ml.SVM_C_SVC
        if index == 1:
            type = ml.SVM_NU_SVC
        elif index == 2:
            type = ml.SVM_ONE_CLASS
        elif index == 3:
            type = ml.SVM_EPS_SVR

        print("Choose kernel:")
        print("0 - LINEAR")
        print("1 - POLY")
        print("2 - RBF")
        #радиальная функция
        index = int(input("Write index [0]: "))
        kernel = ml.SVM_LINEAR
        if index == 1:
            kernel = ml.SVM_POLY
        elif index == 2:
            kernel = ml.SVM_RBF

        c = 1
        if (type in [ml.SVM_C_SVC, ml.SVM_EPS_SVR]):
            c = float(input("Write C [1]: "))
        # регулирующий величину штрафа за то, что некоторые точки выходят за границу разделяющей полосы
        gamma = 2
        if (kernel in [ml.SVM_POLY, ml.SVM_RBF]):
            gamma = float(input("Write gamma [2]: "))

        nu = 0.5
        if (type in [ml.SVM_NU_SVC, ml.SVM_EPS_SVR, ml.SVM_ONE_CLASS]):
            nu = float(input("Write nu [0 --- 1]: "))

        degree = 1
        if (kernel in [ml.SVM_POLY]):
            degree = int(input("Write degree > 0 [1]: "))
        return SVM(type, kernel, c, gamma, nu, degree)

    def createRtree(self):
        n_trees = int(input("Write n trees [25]: "))
        min_sample_count = int(input("Write min sample count [10]: "))
        max_depth = int(input("Write max depth [10]: "))
        return RTrees(n_trees, min_sample_count, max_depth)

    def createDtree(self):
        min_sample_count = int(input("Write min sample count [10]: "))
        max_depth = int(input("Write max depth [10]: "))
        return DTrees(min_sample_count, max_depth)

    def read_file(self, filename):
        fs_read = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
        self.features_train = fs_read.getNode('features_train').mat()
        self.response_train = fs_read.getNode('response_train').mat()
        self.features_test = fs_read.getNode('features_test').mat()
        self.response_test = fs_read.getNode('response_test').mat()
        fs_read.release()


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

    def predict(self, values):
        result = self.model.predict(values)
        return np.ravel(result[1])

    def error(self, features, responses):
        _responses = np.array(np.ravel(self.predict(features)), dtype=np.int)
        responses = np.ravel(responses)
        return 1 - np.sum(responses == _responses) / len(responses)


class SVM(StatModel):
    '''wrapper for OpenCV SimpleVectorMachine algorithm'''

    def __init__(self, type=ml.SVM_C_SVC, kernel=ml.SVM_LINEAR, c=2., gamma=1., nu=0.5, degree=1):
        self.model = ml.SVM_create()
        self.model.setType(type)
        self.model.setKernel(kernel)
        self.model.setC(c)
        self.model.setGamma(gamma)
        self.model.setNu(nu)
        self.model.setDegree(degree)

    def train(self, samples, responses):
        # setting algorithm parameters
        return self.model.train(samples, ml.ROW_SAMPLE, responses)


class DTrees(StatModel):
    '''wrapper for OpenCV SimpleVectorMachine algorithm'''

    def __init__(self, min_sample_count=1, max_depth=10):
        # self.model = cv2.DTrees_create()
        self.model = ml.DTrees_create()

        self.model.setMinSampleCount(min_sample_count)
        self.model.setCVFolds(1)
        self.model.setMaxDepth(max_depth)

    def train(self, samples, responses):
        # setting algorithm parameters
        return self.model.train(samples, ml.ROW_SAMPLE, responses)


class RTrees(StatModel):
    '''wrapper for OpenCV SimpleVectorMachine algorithm'''

    def __init__(self, n_trees=10, min_sample_count=2, max_depth=10):
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
        self.model = ml.GBTrees_create()
        eps = 1
        criteria = (cv2.TERM_CRITERIA_MAX_ITER, n_trees, eps)
        self.model.setTermCriteria(criteria)
        self.model.setMinSampleCount(min_sample_count)
        self.model.setMaxDepth(max_depth)

    def train(self, samples, responses):
        # setting algorithm parameters
        self.model.setMaxCategories(len(np.unique(responses)))
        return self.model.train(samples, ml.ROW_SAMPLE, responses)


app = Application()
app.read_file("clusterization/dataset3.yml")
app.start()
# dataset2 SVM(type=ml.SVM_ONE_CLASS, kernel=ml.SVM_RBF)
# cvf = DTrees(max_depth=54)
# cvf.train(app.features_train, app.response_train)
# cvf.drawPoints(app.features_train, app.response_train)
