import cv2
import sys

# video_capture = cv2.VideoCapture(0)
# isGray = len(sys.argv) > 1 and sys.argv[1] == "gray"

img = cv2.imread('logo.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imwrite("opencv-logo-gray.png",gray)

ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
cv2.imwrite("opencv-logo-contour.png", img)

#
# while True:
#     # Capture frame-by-frame
#     _, frame = video_capture.read()
#
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
#     im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
#     cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
#
#     cv2.imshow('Video', gray if isGray else frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # When everything is done, release the capture
# video_capture.release()
# cv2.destroyAllWindows()
