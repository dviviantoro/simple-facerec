import itertools
from deepface import DeepFace
import numpy as np
import time

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

# 1. Load embeddings
def load_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split(': ')
                if len(parts) == 2:
                    file_name, embedding_str = parts
                    embedding = list(map(float, embedding_str.split()))
                    embeddings[file_name] = embedding
    return embeddings

# 2. Group similar faces into clusters
def verify_and_cluster(embeddings, threshold):
    """
    Simulates clustering using DeepFace.verify to compare all face pairs.
    """
    clusters = []  # List of clusters (each cluster is a list of similar files)
    visited = set()  # Keep track of already checked files
    
    for file_a in embeddings.keys():
        if file_a in visited:
            continue
        
        # Start a new cluster with the current file
        current_cluster = [file_a]
        visited.add(file_a)
        
        for file_b in embeddings.keys():
            if file_b != file_a and file_b not in visited:
                # Verify if the two files are similar

                print(file_a)
                print(file_b)

                # result = DeepFace.verify(file_a, file_b, distance_metric="cosine", model_name="ArcFace", detector_backend="retina_face")
                result = DeepFace.verify(embeddings[file_a],
                                         embeddings[file_b],
                                         distance_metric="cosine",
                                         model_name="ArcFace",
                                         detector_backend="retinaface",
                                         threshold=threshold,
                                         silent=True)
                
                if result["verified"]:
                    current_cluster.append(file_b)
                    visited.add(file_b)
        
        clusters.append(current_cluster)
    
    return clusters

# 3. Main function
if __name__ == "__main__":
    start_time = time.time()


    file_path = "/home/pi5/simple-facerec/embeddings-arcface.txt"  # Path to your embeddings file
    embeddings = load_embeddings(file_path)

    # Group faces into clusters
    clusters = verify_and_cluster(embeddings, 0.07)

    # Print clusters
    print("Clusters:")
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i}: {cluster}")

    result = DeepFace.verify(embeddings["Image_face0002.jpg"], embeddings["Image_face0003.jpg"], distance_metric="cosine", model_name="ArcFace", detector_backend="retina_face", threshold=0.05)
    print(result)


    print("--- %s seconds ---" % (time.time() - start_time))