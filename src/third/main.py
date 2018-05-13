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

    def drawPartition(self, img=None, colors=[(255, 0, 0), (0, 0, 255)]):
        width = 200
        height = 200
        img = img if img is not None else np.zeros((width, height, 3), np.uint8)

        coordinates = np.array([[x, y] for x in range(width) for y in range(height)], dtype=np.float32)

        predicts = self.predict(coordinates)
        for x in range(width):
            for y in range(height):
                index = y * height + x
                img[x][y] = colors[int(predicts[index])]
        return img

    def drawPoints(self, points, responses, img=None, colors=[(255, 0, 0), (0, 0, 255)]):
        width = 200
        height = 200
        img = img if img is not None else np.zeros((width, height, 3), np.uint8)

        for i in range(len(points)):
            point = (int(points[i][0]), int(points[i][1]))
            cv2.circle(img, point, 2, colors[int(responses[i])], -1)
            cv2.circle(img, point, 3, (255, 255, 255), 0)
        return img

    def draw(self, img):
        plt.imshow(img)
        plt.gca().invert_yaxis()
        plt.show()

    def predict(self, values):
        pass


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

    def predict(self, values):
        result = self.model.predict(values)
        return np.reshape(result[1], len(result[1]))


#
samples = np.array([[40, 10], [40, 40], [20, 40], [20, 20]], dtype=np.float32)
y_train = np.array([0, 0, 0, 1], dtype=np.int)
clf = SVM()

clf.train(samples, y_train)
y_val = clf.predict(samples)  # clf.save("file.dat")

img = clf.drawPartition()
img = clf.drawPoints(samples, y_val, img)
clf.draw(img)

# Generate data and labels
# trainData =
# trainLabels = np.array([0, 1, 0, 1], dtype=np.int)
#
# # Create SVM
# svm = cv2.ml.SVM_create()
# svm.setType(cv2.ml.SVM_C_SVC)
# svm.setKernel(cv2.ml.SVM_RBF)
# svm.setC(2)
# svm.setGamma(2)
#
# # Train : error occurs here.
# svm.train(trainData, cv2.ml.ROW_SAMPLE, trainLabels)
