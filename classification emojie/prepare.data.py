 import os
import cv2











 









from util import get_face_landmarks


data_dir = './data'

for emotion in os.listdir(data_dir):
    for image_path_ in os.listdir(os.path.join(data_dir, emotion)):
        image_path = os.path.join(data_dir, emotion, image_path_)
        image = cv2.imread(image_path)

        face_landmarks = get_face_landmarks(image)
        print(face_landmarks)