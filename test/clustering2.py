from deepface import DeepFace
from sklearn.cluster import DBSCAN
import numpy as np
import os
import time
from dotenv import load_dotenv
load_dotenv()
cwd = os.getenv("CWD")


# Set the directory for the images (change to your own directory)
image_folder = f'{cwd}/archive/2024-12-16id374/faces'
image_files = [f for f in os.listdir(image_folder) if f.endswith(('jpg', 'jpeg', 'png'))]
image_files.sort()

start_time = time.time()
# Extract face embeddings using DeepFace
embeddings = []
for img_file in image_files:
    img_path = os.path.join(image_folder, img_file)
    try:
        print(f"Processing image embedding: {img_file}")
        # Extracting the embeddings (this returns a list of dictionaries, we need the embedding array)
        embedding = DeepFace.represent(img_path, model_name="ArcFace", detector_backend="retinaface", enforce_detection=False)[0]['embedding']
        embeddings.append(embedding)
    except Exception as e:
        print(f"Error processing image {img_file}: {e}")
    print("--- %s seconds ---" % (time.time() - start_time))

# Convert embeddings list to a numpy array
embeddings = np.array(embeddings)

output_file = 'embeddings.txt'
with open(output_file, 'w') as f:
    for i, embedding in enumerate(embeddings):
        # Convert each embedding (numpy array) to a space-separated string
        embedding_str = ' '.join(map(str, embedding))
        f.write(f"Image_{image_files[i]}: {embedding_str}\n")

print(f"Embeddings saved to {output_file}")

# Apply DBSCAN clustering
db = DBSCAN(eps=0.2, min_samples=4, metric='euclidean')  # Adjust eps and min_samples as needed
cluster_labels = db.fit_predict(embeddings)

# Get the number of unique clusters (excluding noise labeled as -1)
unique_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
print(f"Number of unique faces (clusters): {unique_clusters}")

# Create a dictionary to store images by their cluster labels
clusters_dict = {}
for idx, label in enumerate(cluster_labels):
    if label != -1:  # Skip noise points labeled as -1
        if label not in clusters_dict:
            clusters_dict[label] = []
        clusters_dict[label].append(image_files[idx])
        
# Display the images grouped by their respective cluster labels
for label, images in clusters_dict.items():
    print(f"\nCluster {label}:")
    for img in images:
        print(f" - {img}")

print("--- %s seconds ---" % (time.time() - start_time))
