import cv2 as cv
import numpy as np

# Updated training data and labels
# Positive (label=1) inside a rectangular region, negatives (label=-1) outside
labels = np.array([1, 1, -1, -1])
trainingData = np.array([
    [200, 200],  # positive
    [300, 300],  # positive
    [100, 250],  # negative
    [400, 100]   # negative
], dtype=np.float32)

# Train the SVM
svm = cv.ml.SVM_create()
svm.setType(cv.ml.SVM_C_SVC)
svm.setKernel(cv.ml.SVM_LINEAR)
svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
svm.train(trainingData, cv.ml.ROW_SAMPLE, labels)

# Create black image canvas
width, height = 512, 512
image = np.zeros((height, width, 3), dtype=np.uint8)

# Draw decision regions as rectangles
for y in range(height):
    for x in range(width):
        sample = np.array([[x, y]], dtype=np.float32)
        response = svm.predict(sample)[1]
        if response == 1:
            image[y, x] = (200, 255, 200)  # Light green
        else:
            image[y, x] = (200, 200, 255)  # Light blue

# Draw training data points as cyan circles
for point, label in zip(trainingData, labels):
    color = (255, 255, 0)  # Cyan
    cv.circle(image, (int(point[0]), int(point[1])), 6, color, -1)

# Draw support vectors
sv = svm.getUncompressedSupportVectors()
for sv_point in sv:
    x, y = int(sv_point[0]), int(sv_point[1])
    cv.rectangle(image, (x - 8, y - 8), (x + 8, y + 8), (0, 0, 255), 2)

# Save and show
cv.imwrite('svm_output.png', image)
cv.imshow('SVM with Rectangles and Cyan Circles', image)
cv.waitKey()
cv.destroyAllWindows()
