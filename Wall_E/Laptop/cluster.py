import os
import cv2
import numpy as np
import torch
from sklearn.cluster import DBSCAN
from facenet_pytorch import InceptionResnetV1

# Configuration
INPUT_DIR = "unknown_faces"
OUTPUT_DIR = "clustered_unknowns"
EMBEDDING_SIZE = 512  # Facenet output
DBSCAN_EPS = 0.8      # Distance threshold, adjust as needed
DBSCAN_MIN_SAMPLES = 2

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

embeddings = []
img_paths = []

# Step 1: Extract embeddings
for fname in os.listdir(INPUT_DIR):
    path = os.path.join(INPUT_DIR, fname)
    img = cv2.imread(path)
    if img is None:
        print(f"Warning: Cannot read {path}")
        continue
    try:
        img_resized = cv2.resize(img, (160, 160))
    except cv2.error:
        print(f"Warning: Cannot resize {path}")
        continue
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    tensor = torch.tensor(img_rgb / 255., dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = resnet(tensor).cpu().numpy()[0]
    embeddings.append(emb)
    img_paths.append(path)

if len(embeddings) == 0:
    print("No valid images found in the input folder.")
    exit(1)

embeddings = np.array(embeddings)

# Step 2: Cluster
print(f"Clustering {len(embeddings)} images...")
clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric='euclidean').fit(embeddings)
labels = clustering.labels_

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"Found {n_clusters} clusters.")

# Step 3: Save clustered images
for idx, (label, img_path) in enumerate(zip(labels, img_paths)):
    if label == -1:
        cluster_dir = os.path.join(OUTPUT_DIR, "noise")
    else:
        cluster_dir = os.path.join(OUTPUT_DIR, f"cluster_{label}")
    os.makedirs(cluster_dir, exist_ok=True)
    base = os.path.basename(img_path)
    save_path = os.path.join(cluster_dir, base)
    cv2.imwrite(save_path, cv2.imread(img_path))

print("Clustering complete! Clustered images saved in:", OUTPUT_DIR)
