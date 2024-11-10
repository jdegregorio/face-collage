import cv2
import numpy as np
import mediapipe as mp
import logging

mp_face_mesh = mp.solutions.face_mesh

def estimate_head_pose(image_path):
    image = cv2.imread(image_path)
    if image is None:
        logging.warning(f"Image {image_path} is unreadable.")
        return None, None, None
    height, width, _ = image.shape

    # Camera internals
    focal_length = width
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
            return None, None, None

        face_landmarks = results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark

        # 2D image points
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
            (0.0, -110.0, -10.0),        # Chin
            (-70.0, 70.0, -50.0),        # Left eye left corner
            (70.0, 70.0, -50.0),         # Right eye right corner
            (-60.0, -70.0, -50.0),       # Left mouth corner
            (60.0, -70.0, -50.0)         # Right mouth corner
        ])

        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        if not success:
            logging.debug(f"solvePnP failed for {image_path}.")
            return None, None, None

        # Convert rotation vector to Euler angles
        rotation_mat, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_mat)
        yaw = angles[1] * 180.0 / np.pi
        pitch = angles[0] * 180.0 / np.pi
        roll = angles[2] * 180.0 / np.pi

        return yaw, pitch, roll
