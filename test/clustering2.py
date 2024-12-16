import os
import cv2
import numpy as np
from deepface import DeepFace
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Step 1: Load images from a directory
def load_images_from_directory(directory_path):
    images = []
    image_paths = []
    
    for filename in os.listdir(directory_path):
        img_path = os.path.join(directory_path, filename)
        if img_path.endswith(('jpg', 'jpeg', 'png')):
            img = cv2.imread(img_path)
            images.append(img)
            image_paths.append(img_path)
    
    return images, image_paths

# Step 2: Compute face embeddings using DeepFace
def compute_face_embeddings(images):
    embeddings = []
    
    for image in images:
        # Use DeepFace to find the embedding
        # result = DeepFace.represent(image, model_name="VGG-Face", enforce_detection=False)
        result = DeepFace.represent(image, model_name="Facenet", detector_backend="retinaface", enforce_detection=False)
        embeddings.append(result[0]["embedding"])
    
    return np.array(embeddings)

# Step 3: Cluster embeddings using DBSCAN
def cluster_embeddings_dbscan(embeddings, eps=0.5, min_samples=2):
    # DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    dbscan.fit(embeddings)
    return dbscan

# Step 4: Visualize and display the clustered images
def display_clusters(image_paths, labels):
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        if label == -1:
            continue  # -1 means noise, so we skip it
        print(f"Cluster {label}:")
        cluster_images = [image_paths[i] for i in range(len(labels)) if labels[i] == label]
        for img_path in cluster_images:
            print(f"  {img_path}")
            img = cv2.imread(img_path)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.show()

# Main function
def main(directory_path, eps=0.5, min_samples=2):
    # Step 1: Load images
    images, image_paths = load_images_from_directory(directory_path)
    
    # Step 2: Compute embeddings
    embeddings = compute_face_embeddings(images)
    
    # Step 3: Cluster embeddings using DBSCAN
    dbscan = cluster_embeddings_dbscan(embeddings, eps, min_samples)
    labels = dbscan.labels_
    
    # Step 4: Display the clustered images
    display_clusters(image_paths, labels)

# Run the program (use your image directory)
directory_path = '/Users/deny/optimized-facerec/archive/2024-12-15id454/faces_retinaface'  # Replace with the path to your image folder
main(directory_path, eps=0.5, min_samples=2)
