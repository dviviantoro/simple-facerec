import os
import cv2
import time
from deepface import DeepFace
from dotenv import load_dotenv
load_dotenv()
cwd = os.getenv('CWD')

folder_path = f"{cwd}/archive/2024-12-16id374"
backend = ['opencv', 'retinaface', 'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'centerface']

def create_dir():
    try:
        os.makedirs(f"{folder_path}/faces")
    except Exception as e:
        print(e)

def crop_face():
    face_count = 0
    file_list = []
    for i in os.listdir(folder_path):
        file_list.append(f"{folder_path}/{i}")
    file_list.sort()

    for i in file_list:
        print(i)
        try:
            faces = DeepFace.extract_faces(i, detector_backend = backend[1], align= True)
            img = cv2.imread(i)

            for j in faces:
                face_count += 1
                formatted_face = "{:04d}".format(face_count)
                facial_area = j["facial_area"]
                x1 = facial_area["x"]
                y1 = facial_area["y"]
                x2 = facial_area["x"] + facial_area["w"]
                y2 = facial_area["y"] + facial_area["h"]
                cropped_face = img[y1:y2, x1:x2]
                output_filename = f"{folder_path}/faces/face{formatted_face}.jpg"
                cv2.imwrite(output_filename, cropped_face)
        except Exception as e:
            print(e)

start_time = time.time()
create_dir()
crop_face()
print("--- %s seconds ---" % (time.time() - start_time))
