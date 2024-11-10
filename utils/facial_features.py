import cv2
import mediapipe as mp
import logging

mp_face_mesh = mp.solutions.face_mesh

def extract_facial_features(image_path):
    """
    Extracts additional facial features such as eye openness and mouth openness.
    """
    image = cv2.imread(image_path)
    if image is None:
        logging.warning(f"Image {image_path} is unreadable.")
        return None

    height, width, _ = image.shape

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            logging.debug(f"No face landmarks found in {image_path}.")
            return None

        face_landmarks = results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark

        # Calculate eye openness
        left_eye_top = landmarks[159]
        left_eye_bottom = landmarks[145]
        left_eye_openness = abs(left_eye_top.y - left_eye_bottom.y) * height

        right_eye_top = landmarks[386]
        right_eye_bottom = landmarks[374]
        right_eye_openness = abs(right_eye_top.y - right_eye_bottom.y) * height

        avg_eye_openness = (left_eye_openness + right_eye_openness) / 2

        # Calculate mouth openness
        upper_lip = landmarks[13]
        lower_lip = landmarks[14]
        mouth_openness = abs(upper_lip.y - lower_lip.y) * height

        features = {
            'left_eye_openness': left_eye_openness,
            'right_eye_openness': right_eye_openness,
            'avg_eye_openness': avg_eye_openness,
            'mouth_openness': mouth_openness
        }

        return features
