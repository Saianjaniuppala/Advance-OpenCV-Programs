import cv2
import numpy as np

# Load your video
cap = cv2.VideoCapture('C:/Users/hi/OneDrive/Documents/VSgima/Chapter C/slow_traffic_small.mp4')

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Read first frame
ret, frame = cap.read()
if not ret:
    print("Error: Could not read frame.")
    exit()

# Initial tracking window
x, y, w, h = 300, 200, 100, 50
track_window = (x, y, w, h)

# Set up ROI
roi = frame[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Termination criteria
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    ret, track_window = cv2.meanShift(dst, track_window, term_crit)

    x, y, w, h = track_window
    final_img = cv2.rectangle(frame, (x, y), (x+w, y+h), 255, 2)
    cv2.imshow('MeanShift Tracking', final_img)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
