import sys
sys.path.append(".")

import json
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from utils.duplicates import remove_duplicates
from utils.tables import draw_table_tops, people_at_table

VIDEO_PATH = "Cctv-Resturant-Demo.mp4"
TABLES_PATH = "tables.json"
TABLE_TOPS_PATH = "table_tops.json"

DISPLAY_SIZE = (960, 540)
PERSON_CLASS_ID = 0
CONFIDENCE = 0.3

RED = (0, 0, 255)
GREEN = (0, 255, 0)

OCCUPY_SECONDS = 3
FREE_SECONDS = 2

# Speed settings
DETECT_EVERY = 3   # run YOLO on 1 frame out of 3
IMGSZ = 480        # smaller image = faster (try 416 if still slow)

device = 0 if torch.cuda.is_available() else "cpu"

# Load YOLO model (nano = much faster than medium)
model = YOLO("models/yolo11n.pt")

# Load table areas
with open(TABLES_PATH, "r") as f:
    tables = json.load(f)

# Load table tops
with open(TABLE_TOPS_PATH, "r") as f:
    table_tops = json.load(f)

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

# Timers based on video time (not on number of processed frames)
present_since = {name: None for name in tables}
absent_since = {name: None for name in tables}

# Initial occupancy state
occupied = {name: False for name in tables}

frame_idx = 0
boxes_xyxy = []
centers = []

start_time = time.time()

while True:
    # Skip frames if we are late, to keep the original video speed
    video_t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    lag = (time.time() - start_time) - video_t

    if lag > 1 / fps:
        for _ in range(int(lag * fps)):
            cap.grab()

    ok, frame = cap.read()

    if not ok:
        break

    # Current time in the video (seconds)
    t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

    # Detect people only every DETECT_EVERY frames
    if frame_idx % DETECT_EVERY == 0:
        result = model(
            frame,
            classes=[PERSON_CLASS_ID],
            conf=CONFIDENCE,
            imgsz=IMGSZ,
            device=device,
            half=(device != "cpu"),
            verbose=False
        )[0]

        people = remove_duplicates(result.boxes)

        boxes_xyxy = []
        centers = []

        for box in people:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes_xyxy.append((x1, y1, x2, y2))
            centers.append(((x1 + x2) // 2, (y1 + y2) // 2))

    frame_idx += 1

    # Draw the last known detections
    for (x1, y1, x2, y2) in boxes_xyxy:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

    for c in centers:
        cv2.circle(frame, c, 5, (255, 255, 255), -1)

    colors = {}
    occupied_count = 0

    # Check every table
    for name, points in tables.items():

        someone_here = people_at_table(points, centers) > 0

        if someone_here:
            if present_since[name] is None:
                present_since[name] = t
            absent_since[name] = None

            # Mark occupied after OCCUPY_SECONDS
            if t - present_since[name] >= OCCUPY_SECONDS:
                occupied[name] = True

        else:
            if absent_since[name] is None:
                absent_since[name] = t
            present_since[name] = None

            # Mark free after FREE_SECONDS
            if t - absent_since[name] >= FREE_SECONDS:
                occupied[name] = False

        # Choose the table colour
        if occupied[name]:
            colors[name] = RED
            occupied_count += 1
        else:
            colors[name] = GREEN

    # Draw coloured table tops
    frame = draw_table_tops(
        frame,
        table_tops,
        colors
    )

    # Display table names
    for name, points in table_tops.items():
        x, y = np.array(points).mean(axis=0).astype(int)

        cv2.putText(
            frame,
            name,
            (x - 40, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    # Display occupancy counter
    cv2.rectangle(
        frame,
        (10, 10),
        (350, 60),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"Occupied: {occupied_count} / {len(tables)}",
        (20, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2
    )

    # Display video
    cv2.imshow(
        "Phase 2 - Occupancy Timer (press Q to quit)",
        cv2.resize(frame, DISPLAY_SIZE)
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()