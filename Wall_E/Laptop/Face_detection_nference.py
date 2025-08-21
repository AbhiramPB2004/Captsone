import cv2
import os
import pickle
import numpy as np
import torch
import time
from collections import defaultdict, deque, Counter
from facenet_pytorch import MTCNN, InceptionResnetV1

DB_FILE = "faces_db.pkl"
UNKNOWN_CLUSTERS_DIR = "unknown_clusters"
VIDEO_SOURCE = "http://192.168.137.96:5000/video_feed"
os.makedirs(UNKNOWN_CLUSTERS_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

if os.path.exists(DB_FILE):
    with open(DB_FILE, "rb") as f:
        known_face_embeddings, known_face_names = pickle.load(f)
else:
    known_face_embeddings, known_face_names = [], []

# Efficient learning params
EMBEDDING_HISTORY_SIZE = 10
person_buffers = defaultdict(lambda: deque(maxlen=EMBEDDING_HISTORY_SIZE))
for emb, name in zip(known_face_embeddings, known_face_names):
    person_buffers[name].append(emb)
last_learn_time = defaultdict(lambda: 0)
LEARN_INTERVAL = 20  # seconds
EMBEDDING_DIFFERENCE_THRESHOLD = 0.7
FRAME_SKIP = 3
frame_idx = 0

# Tracking & smoothing
HISTORY_LEN = 5
MAX_TRACK_DIST = 50  # pixels
label_histories = []

# Unknown clustering
unknown_clusters = []  # Each: [representative_embedding, cluster_idx]
UNKNOWN_CLUSTER_THRESHOLD = 0.8   # Distance for cluster membership
UNKNOWN_CLUSTER_MAX_SAMPLES = 5   # For each cluster, track up to N representatives

def recognize_face(face_img):
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_pil = torch.tensor(face_rgb / 255., dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = resnet(face_pil).detach().cpu().numpy()
    return embedding[0]

def compare_faces(known_embeddings, face_embedding, threshold=0.8):
    if not known_embeddings:
        return -1
    known_embeddings = np.array(known_embeddings)
    distances = np.linalg.norm(known_embeddings - face_embedding, axis=1)
    min_dist_idx = np.argmin(distances)
    if distances[min_dist_idx] < threshold:
        return min_dist_idx
    return -1

def update_label_histories(detected_boxes, new_labels):
    global label_histories
    new_histories = []
    used_indices = set()
    if not detected_boxes:
        label_histories.clear()
        return
    centroids = [((l+r)//2, (t+b)//2) for l, t, r, b in detected_boxes]
    for old_hist in label_histories:
        old_cx, old_cy = old_hist['centroid']
        dists = [np.hypot(cx-old_cx, cy-old_cy) for cx, cy in centroids]
        if dists and min(dists) < MAX_TRACK_DIST:
            idx = np.argmin(dists)
            used_indices.add(idx)
            hist = old_hist
            hist['centroid'] = centroids[idx]
            hist['buffer'].append(new_labels[idx])
            hist['box'] = detected_boxes[idx]
            new_histories.append(hist)
    for i, (box, label) in enumerate(zip(detected_boxes, new_labels)):
        if i not in used_indices:
            hist = {
                'centroid': ((box[0]+box[1])//2, (box[2]+box[3])//2),
                'box': box,
                'buffer': deque([label], maxlen=HISTORY_LEN)
            }
            new_histories.append(hist)
    label_histories = new_histories

# Unknown clustering helpers
def get_cluster_for_embedding(embedding):
    """Returns the cluster index if close to an existing, else None."""
    if not unknown_clusters:
        return None
    rep_embeddings = [np.mean(np.array(c), axis=0) if isinstance(c, list) else c for c in unknown_clusters]
    rep_embeddings = np.array(rep_embeddings)
    distances = np.linalg.norm(rep_embeddings - embedding, axis=1)
    min_idx = np.argmin(distances)
    if distances[min_idx] < UNKNOWN_CLUSTER_THRESHOLD:
        return unknown_clusters[min_idx][2]
    return None

def add_embedding_to_cluster(embedding, img, cluster_idx):
    """Adds embedding and image to a given cluster."""
    cluster_folder = os.path.join(UNKNOWN_CLUSTERS_DIR, f"unknown_cluster_{cluster_idx}")
    os.makedirs(cluster_folder, exist_ok=True)
    # Save image
    img_path = os.path.join(cluster_folder, f"{int(time.time())}.jpg")
    cv2.imwrite(img_path, img)
    # Update representatives (limit memory for efficiency)
    for cluster in unknown_clusters:
        if cluster[2] == cluster_idx:
            if isinstance(cluster, list):
                cluster.append(embedding)
                if len(cluster) > UNKNOWN_CLUSTER_MAX_SAMPLES:
                    cluster = cluster[-UNKNOWN_CLUSTER_MAX_SAMPLES:]
            else:
                cluster = [cluster, embedding]
            return

def create_new_unknown_cluster(embedding, img):
    """Creates a new cluster folder and adds this sample as representative."""
    cluster_idx = len(unknown_clusters) + 1
    cluster_folder = os.path.join(UNKNOWN_CLUSTERS_DIR, f"unknown_cluster_{cluster_idx}")
    os.makedirs(cluster_folder, exist_ok=True)
    img_path = os.path.join(cluster_folder, f"{int(time.time())}.jpg")
    cv2.imwrite(img_path, img)
    unknown_clusters.append([[embedding], cluster_idx])  # List of embeddings for this cluster

cap = cv2.VideoCapture(VIDEO_SOURCE)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame fetch failed. Reconnecting...")
        cap.release()
        time.sleep(1)
        cap = cv2.VideoCapture(VIDEO_SOURCE)
        continue

    frame_idx += 1

    detected_boxes = []
    detected_labels = []

    if frame_idx % FRAME_SKIP == 0:
        boxes, _ = mtcnn.detect(frame)
        if boxes is not None:
            height, width = frame.shape[:2]
            for box in boxes:
                left = max(0, int(box[0]))
                top = max(0, int(box[2]))
                right = min(width, int(box[1]))
                bottom = min(height, int(box[3]))

                if top >= bottom or left >= right:
                    continue

                face_img = frame[top:bottom, left:right]
                if face_img.size == 0:
                    continue

                try:
                    face_img_resized = cv2.resize(face_img, (160, 160))
                except cv2.error:
                    continue

                face_embedding = recognize_face(face_img_resized)
                match_idx = compare_faces(known_face_embeddings, face_embedding)
                now = time.time()
                if match_idx >= 0:
                    name = known_face_names[match_idx]
                    buf = person_buffers[name]
                    if (now - last_learn_time[name] > LEARN_INTERVAL and
                        not any(np.allclose(face_embedding, emb) for emb in buf)):
                        buf.append(face_embedding)
                        last_learn_time[name] = now
                        print(f"[{time.strftime('%H:%M:%S')}] Learned new embedding for {name}. Buffer size: {len(buf)}")
                        # Optionally, prune_database() here if need be
                else:
                    name = "Unknown"
                    # CLUSTERING LOGIC
                    existing_cluster = get_cluster_for_embedding(face_embedding)
                    if existing_cluster is not None:
                        add_embedding_to_cluster(face_embedding, face_img_resized, existing_cluster)
                        print(f"Added to unknown_cluster_{existing_cluster}.")
                    else:
                        create_new_unknown_cluster(face_embedding, face_img_resized)
                        print(f"Created new unknown cluster: {len(unknown_clusters)}.")

                detected_boxes.append([left, top, right, bottom])
                detected_labels.append(name)

        update_label_histories(detected_boxes, detected_labels)

    # Draw from label histories (smooth flicker)
    for hist in label_histories:
        l, t, r, b = hist['box']
        label = Counter(hist['buffer']).most_common(1)[0]
        color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (l, t), (r, b), color, 2)
        cv2.putText(frame, str(label), (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
