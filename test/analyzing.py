import os
import time
from deepface import DeepFace
import numpy as np
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
cwd = os.getenv('CWD')

image_folder = f"{cwd}/archive/2024-12-16id374/faces"
backend = ['opencv', 'retinaface', 'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'centerface']

def get_face_list(folder=image_folder):
    image_files = [f for f in os.listdir(folder) if f.endswith(('jpg', 'jpeg', 'png'))]
    image_files.sort()
    return image_files

def analyzing_face():
    results = []
    image_files = get_face_list()
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        try:
            single_data = []
            print(f"Analyzing image: {img_file}")
            result = DeepFace.analyze(img_path, detector_backend=backend[1], enforce_detection=False, actions=["emotion", "age", "gender"])
            
            emotion = result[0]["dominant_emotion"] 
            gender = result[0]["dominant_gender"] 
            age = result[0]["age"]

            single_data.append(img_file)
            single_data.append(emotion)
            single_data.append(gender)
            single_data.append(age)

            results.append(single_data)
        except Exception as e:
            print(f"Error analyzing image {img_file}: {e}")
        print("--- %s seconds ---" % (time.time() - start_time))
    
    # print(results)
    results = np.array(results)
    # print(results)
    return results

if __name__ == "__main__":
    start_time = time.time()

    df = pd.DataFrame(analyzing_face())
    df.columns = ["filename", "emotion", "gender", "age"]
    df.to_csv("analyzer.csv")

    print("--- %s seconds ---" % (time.time() - start_time))
