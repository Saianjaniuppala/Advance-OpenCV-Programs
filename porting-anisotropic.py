import cv2
import numpy as np

def calc_gst(input_img, w):
    """Calculate Gradient Structure Tensor using standard OpenCV"""
    img = input_img.astype(np.float32)

    # Compute gradients
    dx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

    # Structure tensor components
    dx2 = dx * dx
    dy2 = dy * dy
    dxdy = dx * dy

    # Smooth components
    J11 = cv2.boxFilter(dx2, -1, (w, w))
    J22 = cv2.boxFilter(dy2, -1, (w, w))
    J12 = cv2.boxFilter(dxdy, -1, (w, w))

    # Eigenvalue calculations
    tmp1 = J11 + J22
    tmp2 = J11 - J22
    tmp2_sq = tmp2 ** 2
    J12_sq = J12 ** 2
    tmp4 = np.sqrt(tmp2_sq + 4.0 * J12_sq)

    lambda1 = 0.5 * (tmp1 + tmp4)
    lambda2 = 0.5 * (tmp1 - tmp4)

    # Final outputs
    lambda_sum = lambda1 + lambda2
    lambda_diff = lambda1 - lambda2

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        coherency = np.true_divide(lambda_diff, lambda_sum)
        coherency[lambda_sum == 0] = 0

    orientation = 0.5 * np.arctan2(2 * J12, J22 - J11) * (180.0 / np.pi)

    return coherency, orientation

def main():
    # Parameters
    WINDOW_SIZE = 52
    COHERENCY_THRESH = 0.43
    ORIENTATION_LOW = 35
    ORIENTATION_HIGH = 57

    # Load image
    img_path = r'C:\Users\hi\OneDrive\Documents\VSgima\Chapter C\opencv-4.x\samples\data\butterfly.jpg'
    img_in = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img_in is None:
        print(f"Error: Could not read image from {img_path}")
        return

    # Compute GST
    coherency, orientation = calc_gst(img_in, WINDOW_SIZE)

    # Thresholding
    coherency_bin = (coherency > COHERENCY_THRESH).astype(np.uint8) * 255
    orientation_bin = cv2.inRange(orientation, ORIENTATION_LOW, ORIENTATION_HIGH)
    segmented = cv2.bitwise_and(coherency_bin, orientation_bin)

    # Blending
    result = cv2.addWeighted(img_in, 0.5, segmented, 0.5, 0.0)

    # Normalize outputs for display
    norm_coherency = cv2.normalize(coherency, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    norm_orientation = cv2.normalize(orientation, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Display
    cv2.imshow('Original', img_in)
    cv2.imshow('Result', result)
    cv2.imshow('Coherency', norm_coherency)
    cv2.imshow('Orientation', norm_orientation)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
