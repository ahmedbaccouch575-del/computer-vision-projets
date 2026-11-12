import cv2
from PIL import Image
import numpy as np

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lowerLimit1 = np.array([0, 100, 100], dtype=np.uint8)
    upperLimit1 = np.array([10, 255, 255], dtype=np.uint8)

    lowerLimit2 = np.array([170, 100, 100], dtype=np.uint8)
    upperLimit2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsvImage, lowerLimit1, upperLimit1)
    mask2 = cv2.inRange(hsvImage, lowerLimit2, upperLimit2)
    mask = cv2.bitwise_or(mask1, mask2)

    mask_ = Image.fromarray(mask)
    bbox = mask_.getbbox()

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    print(bbox)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)

    key = cv2.waitKey(1) & 0xFF
    if key != 255:
        print(f"Touche pressée: {key} (caractère: {chr(key) if key < 128 else '?'})")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()