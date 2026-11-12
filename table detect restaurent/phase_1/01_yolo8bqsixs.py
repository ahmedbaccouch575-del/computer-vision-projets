from ultralytics import YOLO

VIDEO_PATH="C:\table detect restaurent\Cctv-Resturant-Demo.mp4"
DISPLY_SIZE=(960,540)

model=YOLO("models/yolo11l.pt")

cap=cv2.VideoCapture(VIDEO_PATH)

while True:
    # Read the next frame of the video
    ok,frame=cap.read()
    if not ok:
        break
    results=model(frame,verbose=False)
    result=results[0]


    annotated=result.plot()
    annonated_resized=cv2.resize(annotated,DISPLY_SIZE)
