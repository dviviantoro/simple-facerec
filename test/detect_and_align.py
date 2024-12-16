import cv2
import json
from deepface import DeepFace

img_path = "/Users/deny/optimized-facerec/assets/liverpool.jpg"
faces = DeepFace.extract_faces(img_path, detector_backend = "retinaface", align= True)
img = cv2.imread(img_path)
face_count = 0

for i in faces:
    face_count += 1
    formatted_face = "{:04d}".format(face_count)
    facial_area = i["facial_area"]
    x1 = facial_area["x"]
    y1 = facial_area["y"]
    x2 = facial_area["x"] + facial_area["w"]
    y2 = facial_area["y"] + facial_area["h"]
    cropped_face = img[y1:y2, x1:x2]
    output_filename = f"/Users/deny/optimized-facerec/temp/face{formatted_face}.jpg"
    cv2.imwrite(output_filename, cropped_face)