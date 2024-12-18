from deepface import DeepFace
from sklearn.cluster import DBSCAN
import numpy as np
import os
import time
from dotenv import load_dotenv
load_dotenv()
cwd = os.getenv("CWD")


def get_img_list(image_folder):
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('jpg', 'jpeg', 'png'))]
    return image_files.sort()

def extract_embedding(image_folder, image_files):
    embeddings = []
    start_time = time.time()
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        try:
            print(f"Processing image embedding: {img_file}")
            embedding = DeepFace.represent(img_path, model_name="ArcFace", detector_backend="retinaface", enforce_detection=False)[0]['embedding']
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error processing image {img_file}: {e}")
        print("--- %s seconds ---" % (time.time() - start_time))
    
    embeddings = np.array(embeddings)
    return embeddings

def save_to_file(output_file, embeddings, image_files):
    output_file = 'embeddings.txt'
    with open(output_file, 'w') as f:
        for i, embedding in enumerate(embeddings):
            embedding_str = ' '.join(map(str, embedding))
            f.write(f"Image_{image_files[i]}: {embedding_str}\n")
    print(f"Embeddings saved to {output_file}")


if __name__ == "__main__":
    start_time = time.time()
    
    image_folder = "dari redis"
    image_files = get_img_list(image_folder)
    embeddings = extract_embedding(image_folder, image_files)
    save_to_file("output.txt", embeddings, image_files)

    print("--- %s seconds ---" % (time.time() - start_time))
