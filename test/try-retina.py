import cv2
from retinaface import RetinaFace
import json
from deepface import DeepFace

img_path = "/Users/deny/optimized-facerec/assets/liverpool.jpg"
faces = RetinaFace.detect_faces(img_path)
list_data = list(faces.items())
#print(faces)

# Open and read the JSON file
# with open('/Users/deny/optimized-facerec/assets/facial.json', 'r') as file:
#     data = json.load(file)
# list_data = list(data.items())
# print(list_data[1]["facial_area"])

img = cv2.imread(img_path)
for i in list_data:
    facial_area = i[1]["facial_area"]
    cv2.rectangle(img, (facial_area[2], facial_area[3]), (facial_area[0], facial_area[1]), (255, 255, 255), 1)

cv2.imwrite("outputimage1.jpg", img)

DeepFace.re
embeddings = []
for img_path in image_paths:
    embedding = DeepFace.represent(img_path, model_name="VGG-Face", enforce_detection=False)
    embeddings.append(embedding[0]['embedding'])
