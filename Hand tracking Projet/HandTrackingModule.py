import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class handDetector():
    def __init__(self, model_path='hand_landmarker.task', maxHands=2, detectionCon=0.5):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=maxHands,
            min_hand_detection_confidence=detectionCon
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.results = None

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        self.results = self.detector.detect(mp_image)

        if draw and self.results.hand_landmarks:
            h, w, c = img.shape
            for hand_landmarks in self.results.hand_landmarks:
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results and self.results.hand_landmarks:
            if handNo < len(self.results.hand_landmarks):
                h, w, c = img.shape
                for id, lm in enumerate(self.results.hand_landmarks[handNo]):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
        return lmList