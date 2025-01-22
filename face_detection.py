import face_recognition
import cv2
import numpy as np
import os
from collections import deque
import time

ip = "http://192.168.26.22:1234/video"
ip1 = "http://192.0.0.4:8080/video"
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Error: Unable to open video source")
    exit()

folder_path = "./photos"
known_face_encodings = []
known_face_names = []

for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    if os.path.isfile(file_path):
        image = face_recognition.load_image_file(file_path)
        try:
            encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(encoding)
            known_face_names.append(file_name)
        except IndexError:
            print(f"Warning: No face detected in {file_name}. Skipping.")

print(f"Loaded {len(known_face_encodings)} known face encodings.")

face_tracking_history = deque(maxlen=5)
match_start_time = None
matched_face_name = None
REQUIRED_DURATION = 1

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    detected_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances) if face_distances.size > 0 else None

        if best_match_index is not None and matches[best_match_index]:
            detected_names.append(known_face_names[best_match_index])

    face_tracking_history.append(detected_names)
    most_common_name = None

    if face_tracking_history:
        flat_names = [name for sublist in face_tracking_history for name in sublist]
        if flat_names:
            most_common_name = max(set(flat_names), key=flat_names.count)

    if most_common_name:
        if matched_face_name == most_common_name:
            if time.time() - match_start_time >= REQUIRED_DURATION:
                print(f"Confirmed Match: {most_common_name}")
                matched_face_name = None
        else:
            matched_face_name = most_common_name
            match_start_time = time.time()

    for (top, right, bottom, left), name in zip(face_locations, detected_names):
        color = (0, 255, 0) if name == most_common_name else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left + 6, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Live Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
