import cv2
import numpy as np

# ✅ Replace this with the full path to your video file
video_path = "C:/Users/hi/OneDrive/Documents/VSgima/Chapter C/slow_traffic_small.mp4"

# 🎥 Load the video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Failed to open video!")
    exit()

# ⏯ Read the first frame
ret, frame = cap.read()
if not ret or frame is None:
    print("❌ Failed to read the first frame.")
    exit()

# 🖱 Select Region of Interest (ROI)
print("Select a ROI and then press SPACE or ENTER button!")
print("Cancel the selection process by pressing C button!")
r = cv2.selectROI("Select Object to Track", frame, fromCenter=False, showCrosshair=True)
x, y, w, h = r
track_window = (x, y, w, h)

# 🔍 Setup ROI for tracking
roi = frame[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# 🛑 Setup termination criteria: 10 iterations or move by at least 1 point
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

# ▶️ Start tracking
while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # 🌀 Apply CamShift to get the new location
    ret, track_window = cv2.CamShift(dst, track_window, term_crit)

    # 📐 Draw tracking result (rotated rectangle)
    pts = cv2.boxPoints(ret)
    pts = np.intp(pts)  # Updated from np.int0 to np.intp for compatibility
    result = cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

    cv2.imshow("CamShift Tracking", result)

    key = cv2.waitKey(30)
    if key == 27 or key == ord('q'):  # ESC or 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
