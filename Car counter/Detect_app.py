from ultralytics import YOLO
import cv2
import cvzone
import math
import time

cap = cv2.VideoCapture("cars.mp4")
cap.set(3, 960)
cap.set(4, 540)

fps_video = cap.get(cv2.CAP_PROP_FPS)
frame_time = 1 / fps_video

model = YOLO("../Yolo-Weights/yolov8s.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book",
              "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
              ]

classes_to_detect = [2, 3]

while True:
    start_time = time.time()

    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True, conf=0.4, classes=classes_to_detect)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Rectangle plus mince : l (longueur des coins), t (épaisseur des coins), rt (épaisseur du rectangle)
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, t=2, rt=1, colorR=(255, 0, 255))

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)),
                                scale=1, thickness=1, offset=3)

    cv2.imshow("Image", img)

    elapsed = time.time() - start_time
    delay = max(1, int((frame_time - elapsed) * 1000))

    key = cv2.waitKey(delay)
    if key & 0xFF == ord('q'):
        break

    if elapsed > frame_time:
        frames_to_skip = int(elapsed / frame_time)
        for _ in range(frames_to_skip):
            cap.grab()

cap.release()
cv2.destroyAllWindows()