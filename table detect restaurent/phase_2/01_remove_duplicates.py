import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

# Main project folder (parent of phase_2), so "utils" can be found
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from utils.duplicates import remove_duplicates

VIDEO_PATH = str(ROOT / "Cctv-Resturant-Demo.mp4")
MODEL_PATH = str(ROOT / "models" / "yolov8n.pt")
DISPLAY_SIZE = (960, 540)
PERSON_CLASS_ID = 0
CONFIDENCE = 0.3
TOTAL_SEATS = 24  # 6 tables x 4 chairs

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    result = model(frame, classes=[PERSON_CLASS_ID], conf=CONFIDENCE, verbose=False)[0]

    # Remove duplicate boxes, then count only the remaining people
    people = remove_duplicates(result.boxes)
    people_count = len(people)

    for box in people:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"person {confidence:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # People counter
    cv2.putText(frame, f"People: {people_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Capacity bar
    percent_full = min(people_count / TOTAL_SEATS, 1.0) * 100
    bar_x1, bar_y1, bar_x2, bar_y2 = 20, 62, 420, 82
    fill_x2 = bar_x1 + int((bar_x2 - bar_x1) * percent_full / 100)
    cv2.rectangle(frame, (bar_x1, bar_y1), (fill_x2, bar_y2), (0, 255, 0), -1)
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (255, 255, 255), 2)
    cv2.putText(frame, f"{percent_full:.0f}% full", (bar_x2 + 10, bar_y2 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Part 2 - Remove duplicates (press Q to quit)",
               cv2.resize(frame, DISPLAY_SIZE))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()