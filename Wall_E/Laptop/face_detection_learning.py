import cv2
import os
import pickle
import numpy as np
import torch
import time
from collections import defaultdict, deque, Counter
from facenet_pytorch import MTCNN, InceptionResnetV1

DB_FILE = "faces_db.pkl"
UNKNOWN_DIR = "unknown_faces"
VIDEO_SOURCE = "http://192.168.137.96:5000/video_feed"
os.makedirs(UNKNOWN_DIR, exist_ok=True)

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

def is_new_embedding(person_buffer, embedding, threshold=EMBEDDING_DIFFERENCE_THRESHOLD):
    if not person_buffer:
        return True
    distances = np.linalg.norm(np.array(person_buffer) - embedding, axis=1)
    if np.min(distances) > threshold:
        return True
    return False

def prune_database():
    global known_face_embeddings, known_face_names
    known_face_embeddings = []
    known_face_names = []
    for name, buf in person_buffers.items():
        for emb in buf:
            known_face_embeddings.append(emb)
            known_face_names.append(name)
    with open(DB_FILE, "wb") as f:
        pickle.dump((known_face_embeddings, known_face_names), f)

def update_label_histories(detected_boxes, new_labels):
    global label_histories
    new_histories = []
    used_indices = set()
    if not detected_boxes:
        label_histories.clear()
        return

    # Compute centroids for current frame
    centroids = [((l+r)//2, (t+b)//2) for l, t, r, b in detected_boxes]

    # Try to match old histories with new detections (closest centroid)
    for old_hist in label_histories:
        old_cx, old_cy = old_hist['centroid']
        dists = [np.hypot(cx-old_cx, cy-old_cy) for cx, cy in centroids]
        if dists and min(dists) < MAX_TRACK_DIST:
            idx = np.argmin(dists)
            used_indices.add(idx)
            # Update with new label and position
            hist = old_hist
            hist['centroid'] = centroids[idx]
            hist['buffer'].append(new_labels[idx])
            hist['box'] = detected_boxes[idx]
            new_histories.append(hist)

    # For new detections not matched, create new histories
    for i, (box, label) in enumerate(zip(detected_boxes, new_labels)):
        if i not in used_indices:
            hist = {
                'centroid': ((box[0]+box[1])//2, (box[2]+box[3])//2),
                'box': box,
                'buffer': deque([label], maxlen=HISTORY_LEN)
            }
            new_histories.append(hist)

    label_histories = new_histories

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

    # Only process every FRAME_SKIP'th frame for recognition/learning
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
                    # Efficient learning: learn only if new and at interval
                    if (now - last_learn_time[name] > LEARN_INTERVAL and
                        is_new_embedding(buf, face_embedding)):
                        buf.append(face_embedding)
                        last_learn_time[name] = now
                        print(f"[{time.strftime('%H:%M:%S')}] Learned new embedding for {name}. Buffer size: {len(buf)}")
                        prune_database()
                else:
                    name = "Unknown"
                    # Save unknown face
                    unknown_path = os.path.join(
                        UNKNOWN_DIR,
                        f"unknown_{left}_{top}_{right}_{bottom}_{int(now)}.jpg"
                    )
                    cv2.imwrite(unknown_path, face_img_resized)
                    print(f"Unknown face detected and saved as {unknown_path}.")

                detected_boxes.append([left, top, right, bottom])
                detected_labels.append(name)

        # Update label history buffer with this frame's results
        update_label_histories(detected_boxes, detected_labels)

    # Draw from label histories (smooths flicker)
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
