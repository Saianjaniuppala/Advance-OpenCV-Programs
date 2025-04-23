import cv2
import numpy as np

# Configuration
class Config:
    BSize = 7
    BSigmaCol = 30.0
    BSigmaSp = 30.0
    UnshSigma = 3
    UnshStrength = 1.5
    ConfThresh = 0.5
    ClrGreen = (0, 255, 0)
    WinInput = "Input"
    WinFaceBeautification = "Face Beautification"

def unsharp_mask(img, sigma, strength):
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1 + strength, blurred, -strength, 0)

def face_beautification(img, face_detector, show_boxes=False):
    img_bilat = cv2.bilateralFilter(img, Config.BSize, Config.BSigmaCol, Config.BSigmaSp)
    img_sharp = unsharp_mask(img, Config.UnshSigma, Config.UnshStrength)

    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104, 177, 123))
    face_detector.setInput(blob)
    detections = face_detector.forward()

    h, w = img.shape[:2]
    mask_faces = np.zeros(img.shape[:2], dtype=np.uint8)

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > Config.ConfThresh:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype("int")
            cv2.rectangle(mask_faces, (startX, startY), (endX, endY), 255, -1)
            if show_boxes:
                cv2.rectangle(img, (startX, startY), (endX, endY), Config.ClrGreen, 2)

    # Create inverse mask for background
    mask_inv = cv2.bitwise_not(mask_faces)

    # Blend
    face_region = cv2.bitwise_and(img_bilat, img_bilat, mask=mask_faces)
    sharp_region = cv2.bitwise_and(img_sharp, img_sharp, mask=mask_inv)
    result = cv2.add(face_region, sharp_region)

    return result

def main():
    face_model = r"C:\Users\hi\OneDrive\Documents\VSgima\Chapter C\agender-master\face_net\opencv_face_detector_uint8.pb"
    face_proto = r"C:\Users\hi\OneDrive\Documents\VSgima\Chapter C\agender-master\face_net\opencv_face_detector.pbtxt"

    face_detector = cv2.dnn.readNet(face_model, face_proto)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = face_beautification(frame, face_detector, show_boxes=True)

        cv2.imshow(Config.WinInput, frame)
        cv2.imshow(Config.WinFaceBeautification, result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
