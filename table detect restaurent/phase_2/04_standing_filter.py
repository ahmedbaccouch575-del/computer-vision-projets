import sys
sys.path.append(".")

import json

import cv2
import numpy as np
from ultralytics  import YOLO
from utils.duplicaltes import  remove_duplicates
from utilis.standing_filter import  remove_standing
from utils.tables import draw_table_tops, people_at_table

VIDEO_PATH=""
TABLES_PATH="tables.json"
DISPLAY_SIZE=(960,540)
CONFIDENCE=0.3ùRED=(0,0,255)
GREEN=(0,255,0)
OCCUPY_SECONDS=1
FREE_SECONDS=1

model=YOLO("models/yolo11s-pose.pt")
with open(TABLES_PATH) as f;
    tables=json.load(f)
with open(TABLE_TOPS_PATH) as f:
    table_tops=json.load(f)
cap=cv2.VideoCapture(VIDEO_PATH)     
# The video has a fixed number of frames per second, so we can measure time by counting frames.
fps = cap.get(cv2.CAP_PROP_FPS)
occupy_frames = OCCUPY_SECONDS * fps
free_frames = FREE_SECONDS * fps

# Two counters for every table.
present_frames = {name: 0 for name in tables}  # frames with someone at the table
empty_frames = {name: 0 for name in tables}    # frames in a row with nobody at the table

while True:
    ok, frame = cap.read()
    if not ok:
        break

    result = model(frame, conf=CONFIDENCE, verbose=False)[0]
    # Drop the people who are standing (waiters, people walking past), then the duplicates.
# Here one pose model does both jobs: it finds the people and tells us who is standing.
people = remove_standing(result.boxes, result)
people = remove_duplicates(people)

centers = []
for box in people:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    centers.append(center)

colors = {}  # table name -> the color of its table top
occupied_count = 0

for name, points in tables.items():
    someone_here = people_at_table(points, centers) > 0