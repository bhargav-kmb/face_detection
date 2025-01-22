import face_recognition
import cv2
import dlib

# Initialize video capture and face landmarks predictor
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Error: Unable to access the camera.")
    exit()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Eye landmark indices (as per dlib's 68 face landmarks model)
LEFT_EYE_LANDMARKS = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_LANDMARKS = [42, 43, 44, 45, 46, 47]

def get_blink_ratio(eye_points, landmarks):
    """
    Calculate the eye aspect ratio (EAR) to detect blinks.
    
    Parameters:
    - eye_points: List of indices for the eye landmarks
    - landmarks: dlib facial landmark predictor output

    Returns:
    - Eye aspect ratio (float)
    """
    # Compute horizontal eye distance
    left_point = landmarks.part(eye_points[0])
    right_point = landmarks.part(eye_points[3])
    horizontal_length = ((right_point.x - left_point.x)**2 + (right_point.y - left_point.y)**2) ** 0.5

    # Compute vertical eye distances
    top_point = ((landmarks.part(eye_points[1]).x + landmarks.part(eye_points[2]).x) // 2,
                 (landmarks.part(eye_points[1]).y + landmarks.part(eye_points[2]).y) // 2)
    bottom_point = ((landmarks.part(eye_points[5]).x + landmarks.part(eye_points[4]).x) // 2,
                    (landmarks.part(eye_points[5]).y + landmarks.part(eye_points[4]).y) // 2)
    vertical_length = ((bottom_point[0] - top_point[0])**2 + (bottom_point[1] - top_point[1])**2) ** 0.5

    return vertical_length / horizontal_length

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray_frame)

    for face in faces:
        landmarks = predictor(gray_frame, face)
        
        left_eye_ratio = get_blink_ratio(LEFT_EYE_LANDMARKS, landmarks)
        right_eye_ratio = get_blink_ratio(RIGHT_EYE_LANDMARKS, landmarks)

        # Blink detection threshold
        BLINK_THRESHOLD = 0.2
        if left_eye_ratio < BLINK_THRESHOLD and right_eye_ratio < BLINK_THRESHOLD:
            cv2.putText(frame, "Blink Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Liveness Detection", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video_capture.release()
cv2.destroyAllWindows()
