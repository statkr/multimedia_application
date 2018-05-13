import cv2
import numpy as np
from matplotlib import pyplot as plt


class Application:
    def __init__(self):
        self.load_image("logo.png")

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

    def load_image(self, name):
        self.input = cv2.imread(name)

    def apply_linear_filter(self):
        kernel = np.ones((5, 5), np.float32) / 25
        self.output = cv2.filter2D(self.input, -1, kernel)

    def blur(self):
        kernel_size = int(input("Write kernel size: "))
        self.output = cv2.blur(self.input, (kernel_size, kernel_size))

    def gaussian_blur(self):
        kernel_size = int(input("Write kernel size: "))
        sigma = int(input("sigma: "))
        self.output = cv2.GaussianBlur(self.input, (kernel_size, kernel_size), sigma)

    def median_blur(self):
        kernel_size = int(input("Write kernel size: "))
        self.output = cv2.medianBlur(self.input, kernel_size)

    def erode(self):
        kernel_size = int(input("Write kernel size: "))
        iterations = int(input("Write iterations: "))
        #если в окне все единицы то осталвяет, иначе удаляет
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.output = cv2.erode(self.input, kernel, iterations=iterations)

    def dilate(self):
        kernel_size = int(input("Write kernel size: "))
        iterations = int(input("Write iterations: "))
        # если в окне хотя бы одна единица, то добавит
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.output = cv2.dilate(self.input, kernel, iterations=iterations)

    def sobel(self):
        sobelX = int(input("Write sobel X: "))
        sobelY = int(input("Write sobel Y: "))
        gray = cv2.cvtColor(self.input, cv2.COLOR_BGR2GRAY)
        self.output = cv2.Sobel(gray, cv2.CV_64F, sobelX, sobelY, ksize=5)

    def laplacian(self):
        gray = cv2.cvtColor(self.input, cv2.COLOR_BGR2GRAY)
        self.output = cv2.Laplacian(gray, cv2.CV_64F)

    def canny(self):
        minV = int(input("Write min value respectively: "))
        maxV = int(input("Write max value respectively: "))

        self.output = cv2.Canny(self.input, minV, maxV)

    def calcHist(self):
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            histr = cv2.calcHist([self.input], [i], None, [256], [0, 256])
            plt.plot(histr, color=col)
            plt.xlim([0, 256])
        self.output = None
        plt.show()

    def equalizeHist(self):
        gray = cv2.cvtColor(self.input, cv2.COLOR_BGR2GRAY)
        self.output = cv2.equalizeHist(gray)

    def show(self):
        if self.output is None:
            return
        plt.subplot(121), plt.imshow(self.input), plt.title('Original')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(self.output), plt.title('Modified')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def applyOperation(self, index):
        if index == 1:
            self.apply_linear_filter()
        elif index == 2:
            self.blur()
        elif index == 3:
            self.median_blur()
        elif index == 4:
            self.gaussian_blur()
        elif index == 5:
            self.erode()
        elif index == 6:
            self.dilate()
        elif index == 7:
            self.sobel()
        elif index == 8:
            self.laplacian()
        elif index == 9:
            self.canny()
        elif index == 10:
            self.calcHist()
        elif index == 11:
            self.equalizeHist()

    def start(self):
        while True:
            index = self.menu()
            if index == 0:
                fileName = input("Write file name: ")
                self.load_image(fileName)
                print("Image read success")
            elif 1 <= index <= 11:
                if not self.input:
                    print("Image not found, please read image first")
                else:
                    self.applyOperation(index)
                    app.show()
            elif index == 12:
                return


app = Application()
app.applyOperation(11)
app.show()
