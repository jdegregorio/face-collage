import cv2
import numpy as np
import mediapipe as mp
import face_recognition
import logging

mp_face_mesh = mp.solutions.face_mesh

def estimate_head_pose_and_facial_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        logging.warning(f"Image {image_path} is unreadable.")
        return None

    height, width, _ = image.shape

    # Camera internals
    focal_length = 1 * width  # Assuming focal length is width in pixels
    center = (width / 2, height / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype='double'
    )
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            logging.debug(f"No face landmarks found in {image_path}.")
            return None

        face_landmarks = results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark

        # Head pose estimation
        image_points = np.array([
            (landmarks[1].x * width, landmarks[1].y * height),     # Nose tip
            (landmarks[152].x * width, landmarks[152].y * height), # Chin
            (landmarks[33].x * width, landmarks[33].y * height),   # Left eye left corner
            (landmarks[263].x * width, landmarks[263].y * height), # Right eye right corner
            (landmarks[61].x * width, landmarks[61].y * height),   # Left mouth corner
            (landmarks[291].x * width, landmarks[291].y * height)  # Right mouth corner
        ], dtype='double')

        # 3D model points
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -63.6, -12.5),         # Chin
            (-43.3, 32.7, -26.0),        # Left eye left corner
            (43.3, 32.7, -26.0),         # Right eye right corner
            (-28.9, -28.9, -24.1),       # Left mouth corner
            (28.9, -28.9, -24.1)         # Right mouth corner
        ])

        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        if not success:
            logging.debug(f"solvePnP failed for {image_path}.")
            return None

        # Convert rotation vector to rotation matrix
        rotation_mat, _ = cv2.Rodrigues(rotation_vector)

        # Compose projection matrix
        proj_matrix = np.hstack((rotation_mat, translation_vector))

        # Decompose projection matrix to get Euler angles
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
        pitch, yaw, roll = euler_angles.flatten()
        pitch = float(pitch)
        yaw = float(yaw)
        roll = float(roll)

        # Extract facial features
        facial_features = {}

        # Eye openness
        left_eye_top = landmarks[159]
        left_eye_bottom = landmarks[145]
        left_eye_openness = float(abs(left_eye_top.y - left_eye_bottom.y) * height)

        right_eye_top = landmarks[386]
        right_eye_bottom = landmarks[374]
        right_eye_openness = float(abs(right_eye_top.y - right_eye_bottom.y) * height)

        avg_eye_openness = (left_eye_openness + right_eye_openness) / 2

        facial_features['left_eye_openness'] = left_eye_openness
        facial_features['right_eye_openness'] = right_eye_openness
        facial_features['avg_eye_openness'] = avg_eye_openness

        # Mouth openness
        upper_lip = landmarks[13]
        lower_lip = landmarks[14]
        mouth_openness = float(abs(upper_lip.y - lower_lip.y) * height)

        facial_features['mouth_openness'] = mouth_openness

        # Rotation angle (assumed to be zero since image was already aligned)
        facial_features['rotation_angle'] = 0.0

        # Scaling factor (assumed to be 1 since image was resized)
        facial_features['scaling_factor'] = 1.0

        # Centering offsets (assumed to be zero)
        facial_features['centering_offsets'] = (0, 0)

        # Actual expansion (if any)
        facial_features['actual_expansion'] = 0.0

        return yaw, pitch, roll, facial_features

def extract_face_embedding(image_path):
    """
    Extract face embedding using face_recognition library.
    """
    try:
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) == 0:
            logging.warning(f"No face encodings found in {image_path}.")
            return None
        return encodings[0]  # Return the first encoding found in the image
    except Exception as e:
        logging.error(f"Error extracting face embedding from {image_path}: {e}")
        return None
