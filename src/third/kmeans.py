import numpy as np
import cv2
from matplotlib import pyplot as plt

cluster_num = 6
# define criteria and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

filename = "classification/dataset4.yml"
fs_read = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
Z = fs_read.getNode('points').mat()
fs_read.release()

ret, label, center = cv2.kmeans(Z, cluster_num, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

colors = ["r", "g", "b", "y", "m", "c"]
# Now separate the data, Note the flatten()
for i in range(cluster_num):
    cluster = Z[label.ravel() == i]
    plt.scatter(cluster[:, 0], cluster[:, 1], c=colors[i])
    plt.scatter(center[:, 0], center[:, 1], s=80, c='y', marker='s')

plt.xlabel('Height'), plt.ylabel('Weight')
plt.show()
