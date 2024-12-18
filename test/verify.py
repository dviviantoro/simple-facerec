from deepface import DeepFace
import numpy as np
import os
import time
from dotenv import load_dotenv
load_dotenv()
cwd = os.getenv("CWD")

start_time = time.time()

def load_face_encodings(file_path):
    filenames = []
    embeddings = []
    
    with open(file_path, 'r') as f:
        for line in f:
            # Split line by colon (filename: embedding)
            parts = line.split(':')
            filename = parts[0].strip()  # Image name
            encoding = np.fromstring(parts[1], sep=' ')  # Convert the rest to a numpy array
            filenames.append(filename)
            embeddings.append(encoding)
    
    return np.array(embeddings), filenames

def check_face(img1, img2):
    result = DeepFace.verify(img1_path=img1, img2_path=img2, model_name="ArcFace", detector_backend="retina_face")
    # print(result)
    # print(result["verified"])
    return result["verified"]

embeddings, filenames = load_face_encodings("/home/pi5/simple-facerec/embeddings-arcface.txt")

cluster = 0
clustered = []
cluster_list = []

for i in range(len(embeddings)):
    cluster += 1
    current_cluster = []

    face1 = embeddings[i]
    face2 = embeddings[i+1]


    for j in range(len(embeddings)):
        if filenames[i] not in clustered:
            clustered.append(filenames[i])

            print(f"check face {i} and {j}")
            check_face(embeddings[i].tolist(), embeddings[j].tolist())


# embedding = DeepFace.represent("/home/pi5/simple-facerec/archive/2024-12-16id374/faces/face0001.jpg", model_name="ArcFace", detector_backend="retinaface", enforce_detection=False)[0]['embedding']
# print(embedding)

# print(embeddings[0])
# print(embeddings[1])
# check_face(embeddings[0].tolist(), embeddings[1].tolist())

print("--- %s seconds ---" % (time.time() - start_time))
