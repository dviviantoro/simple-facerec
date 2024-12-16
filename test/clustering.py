import os
import numpy as np
from deepface import DeepFace
from sklearn.cluster import DBSCAN
import cv2
import time

# Step 1: Load images and extract embeddings using DeepFace (with RetinaFace backend)
def extract_face_embeddings(image_paths):
    embeddings = []
    
    for img_path in image_paths:
        print(f"Extracting embedding for {img_path}")
        try:
            # DeepFace extracts embeddings using RetinaFace as the face detector
            # result = DeepFace.represent(img_path=img_path, model_name="Facenet", enforce_detection=False)
            result = DeepFace.represent(img_path=img_path, model_name="ArcFace", detector_backend="retinaface", enforce_detection=False)
            # print(result)
            embeddings.append(result[0]["embedding"])
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            continue
    
    return np.array(embeddings)

# Step 2: Perform DBSCAN clustering
def perform_dbscan_clustering(embeddings, eps=0.4, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')  # Using 'cosine' as distance metric
    clusters = dbscan.fit_predict(embeddings)
    return clusters

# Step 3: Main function to load images and cluster them
def cluster_faces(image_folder):
    # Get all image paths from the folder
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder).sort() if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    if len(image_paths) == 0:
        print("No images found in the folder.")
        return
    
    print(f"Found {len(image_paths)} images.")
    
    # Step 1: Extract face embeddings
    print("Extracting embeddings...")
    start_time = time.time()
    embeddings = extract_face_embeddings(image_paths)
    
    if len(embeddings) == 0:
        print("No embeddings extracted. Exiting...")
        return
    
    print(f"Embeddings extracted in {time.time() - start_time:.2f} seconds.")
    
    # Step 2: Perform DBSCAN clustering
    print("Performing DBSCAN clustering...")
    clusters = perform_dbscan_clustering(embeddings)
    
    # Output clustering results
    unique_clusters = set(clusters)
    print(f"\nNumber of unique clusters: {len(unique_clusters) - (1 if -1 in unique_clusters else 0)} (excluding noise)")

    # Display images in each cluster
    for cluster_id in unique_clusters:
        if cluster_id == -1:  # -1 indicates noise (unclustered images)
            continue
        
        print(f"\nCluster {cluster_id}:")
        cluster_images = [image_paths[i] for i in range(len(clusters)) if clusters[i] == cluster_id]
        
        for img_path in cluster_images:
            print(f"  {img_path}")
    
    # Optional: Handle noisy images (cluster label -1)
    noise_images = [image_paths[i] for i in range(len(clusters)) if clusters[i] == -1]
    if noise_images:
        print("\nNoise images (not assigned to any cluster):")
        for img_path in noise_images:
            print(f"  {img_path}")

# Example usage
image_folder = "/Users/deny/optimized-facerec/archive/2024-12-15id454/faces"  # Replace with the folder containing your 100 face images
cluster_faces(image_folder)
