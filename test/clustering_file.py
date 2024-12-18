import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import os
import time
from dotenv import load_dotenv
load_dotenv()
cwd = os.getenv("CWD")

start_time = time.time()

# Function to load the face embeddings from the txt file
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

            # print(encoding)
    
    return np.array(embeddings), filenames

# Load the face encodings from your txt file
file_path = '/home/pi5/simple-facerec/embeddings-arcface.txt'  # Path to the txt file containing face embeddings
face_encodings, filenames = load_face_encodings(file_path)

# # Step 1: Standardize the data (important for DBSCAN)
# face_encodings_scaled = StandardScaler().fit_transform(face_encodings)

# # Step 2: Apply DBSCAN clustering
# dbscan = DBSCAN(eps=0.8, min_samples=5, metric='euclidean')
# labels = dbscan.fit_predict(face_encodings_scaled)

# # Step 3: Filter out the noise points (label == -1)
# unique_faces = []
# unique_filenames = []

# # Collect faces belonging to valid clusters (labels != -1)
# for i, label in enumerate(labels):
#     if label != -1:  # Ignore noise points
#         unique_faces.append(face_encodings[i])
#         unique_filenames.append(filenames[i])

# # Convert the unique faces back to a numpy array for further processing (if needed)
# unique_faces = np.array(unique_faces)

# # Step 4: Display the results (Optional)
# print(f"Number of unique faces (excluding noise): {len(unique_faces)}")
# print("Filenames of unique faces:", unique_filenames)

# Step 2: Perform DBSCAN clustering
def perform_dbscan_clustering(embeddings, eps=0.4, min_samples=2):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')  # Using 'cosine' as distance metric
    clusters = dbscan.fit_predict(embeddings)
    return clusters

# Step 3: Main function to load images and cluster them
def cluster_faces():
    # Step 2: Perform DBSCAN clustering
    print("Performing DBSCAN clustering...")
    clusters = perform_dbscan_clustering(face_encodings)
    
    # Output clustering results
    unique_clusters = set(clusters)
    print(f"\nNumber of unique clusters: {len(unique_clusters) - (1 if -1 in unique_clusters else 0)} (excluding noise)")

    # Display images in each cluster
    for cluster_id in unique_clusters:
        if cluster_id == -1:  # -1 indicates noise (unclustered images)
            continue
        
        print(f"\nCluster {cluster_id}:")
        cluster_images = [filenames[i] for i in range(len(clusters)) if clusters[i] == cluster_id]
        
        for img_path in cluster_images:
            print(f"  {img_path}")
    
    # Optional: Handle noisy images (cluster label -1)
    noise_images = [filenames[i] for i in range(len(clusters)) if clusters[i] == -1]
    if noise_images:
        print("\nNoise images (not assigned to any cluster):")
        for img_path in noise_images:
            print(f"  {img_path}")

cluster_faces()






print("--- %s seconds ---" % (time.time() - start_time))
