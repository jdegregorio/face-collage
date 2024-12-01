import os
import cv2
import mediapipe as mp
import logging
from config import IMAGE_SIZE, DESIRED_FACE_SIZE_RATIO, PROCESSED_IMAGES_DIR, APPLY_YAW_CORRECTION
from utils.face import Face
import uuid
import numpy as np
import math

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

def process_single_image(photo):
    image_path = photo.original_image_path
    processed_images_dir = PROCESSED_IMAGES_DIR

    image = cv2.imread(image_path)
    if image is None:
        logging.warning(f"Image {photo.filename} is unreadable.")
        return None
    height, width, _ = image.shape

    # Convert image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Face detection
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        detection_results = face_detection.process(image_rgb)

    if not detection_results.detections:
        logging.debug(f"No faces detected in {photo.filename}.")
        return None  # No faces detected

    faces = []
    for idx, detection in enumerate(detection_results.detections):
        # Use the detection bounding box to get the region of interest
        bbox = detection.location_data.relative_bounding_box
        x_min = int(bbox.xmin * width)
        y_min = int(bbox.ymin * height)
        bbox_width = int(bbox.width * width)
        bbox_height = int(bbox.height * height)

        # Expand the bounding box slightly to include more of the face
        expansion_ratio = 0.5  # 50% expansion
        x_min_expanded = max(int(x_min - bbox_width * expansion_ratio / 2), 0)
        y_min_expanded = max(int(y_min - bbox_height * expansion_ratio / 2), 0)
        x_max_expanded = min(int(x_min + bbox_width * (1 + expansion_ratio)), width)
        y_max_expanded = min(int(y_min + bbox_height * (1 + expansion_ratio)), height)

        # Crop the face region
        face_region = image_rgb[y_min_expanded:y_max_expanded, x_min_expanded:x_max_expanded]

        # Facial landmarks for alignment and scaling
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
            face_results = face_mesh.process(face_region)

        if not face_results.multi_face_landmarks:
            logging.debug(f"No face landmarks found in face {idx} of {photo.filename}.")
            continue  # Skip this face

        face_landmarks = face_results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark

        # Get landmark coordinates relative to the face_region
        face_region_height, face_region_width = face_region.shape[:2]
        landmarks_array = np.array([[lm.x * face_region_width, lm.y * face_region_height] for lm in landmarks])

        # Calculate key points
        left_eye = landmarks_array[33]
        right_eye = landmarks_array[263]
        eye_center = (left_eye + right_eye) / 2
        nose_tip = landmarks_array[1]
        mouth_center = (landmarks_array[13] + landmarks_array[14]) / 2

        # Calculate rotation angle
        eye_delta = right_eye - left_eye
        angle = np.arctan2(eye_delta[1], eye_delta[0])
        angle_degrees = np.degrees(angle)

        # Rotate the entire image around the center of the face region
        rotation_center = (x_min_expanded + eye_center[0], y_min_expanded + eye_center[1])
        rotation_matrix = cv2.getRotationMatrix2D(rotation_center, angle_degrees, 1)
        rotated_image = cv2.warpAffine(image_rgb, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)

        # Update landmarks after rotation
        ones = np.ones(shape=(len(landmarks_array), 1))
        landmarks_homogenous = np.hstack([landmarks_array + [x_min_expanded, y_min_expanded], ones])
        rotated_landmarks = rotation_matrix.dot(landmarks_homogenous.T).T

        # Recalculate key points after rotation
        left_eye_rotated = rotated_landmarks[33]
        right_eye_rotated = rotated_landmarks[263]
        eye_center_rotated = (left_eye_rotated + right_eye_rotated) / 2
        nose_tip_rotated = rotated_landmarks[1]
        mouth_center_rotated = (rotated_landmarks[13] + rotated_landmarks[14]) / 2

        # Estimate yaw angle
        # Using 3D model points and image points for head pose estimation
        image_points = np.array([
            nose_tip_rotated,                   # Nose tip
            rotated_landmarks[152],             # Chin
            left_eye_rotated,                   # Left eye left corner
            right_eye_rotated,                  # Right eye right corner
            rotated_landmarks[61],              # Left mouth corner
            rotated_landmarks[291]              # Right mouth corner
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

        # Camera internals
        focal_length = width
        center = (width / 2, height / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype='double'
        )
        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        if not success:
            logging.debug(f"Head pose estimation failed for face {idx} in {photo.filename}.")
            continue  # Skip this face

        # Convert rotation vector to Euler angles
        rotation_mat, _ = cv2.Rodrigues(rotation_vector)
        proj_matrix = np.hstack((rotation_mat, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
        pitch, yaw, roll = euler_angles.flatten()
        pitch = float(pitch)
        yaw = float(yaw)
        roll = float(roll)

        # Adjust eye distance based on yaw angle
        if APPLY_YAW_CORRECTION:
            yaw_radians = np.radians(yaw)
            corrected_eye_distance = np.linalg.norm(left_eye_rotated - right_eye_rotated) / np.cos(yaw_radians)
        else:
            corrected_eye_distance = np.linalg.norm(left_eye_rotated - right_eye_rotated)

        # Define desired face size based on corrected inter-eye distance
        desired_face_width = corrected_eye_distance * DESIRED_FACE_SIZE_RATIO

        # Calculate crop box centered on the eye-line, with eye-line centered vertically
        half_face_width = desired_face_width / 2
        x_min_crop = int(eye_center_rotated[0] - half_face_width)
        x_max_crop = int(eye_center_rotated[0] + half_face_width)

        # For y-axis, center the eye-line vertically
        y_center_crop = eye_center_rotated[1]
        y_min_crop = int(y_center_crop - half_face_width)
        y_max_crop = int(y_center_crop + half_face_width)

        # Ensure the crop box is within image bounds
        x_min_crop = max(x_min_crop, 0)
        y_min_crop = max(y_min_crop, 0)
        x_max_crop = min(x_max_crop, width)
        y_max_crop = min(y_max_crop, height)

        # Adjust the crop size if necessary to maintain square aspect ratio
        crop_width = x_max_crop - x_min_crop
        crop_height = y_max_crop - y_min_crop
        min_crop_size = min(crop_width, crop_height)
        x_max_crop = x_min_crop + min_crop_size
        y_max_crop = y_min_crop + min_crop_size

        # Crop the aligned face
        cropped_aligned_face = rotated_image[y_min_crop:y_max_crop, x_min_crop:x_max_crop]

        # Resize to IMAGE_SIZE without distortion
        final_face_image = cv2.resize(cropped_aligned_face, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

        # Generate unique ID for face
        face_id = str(uuid.uuid4())

        # Save processed face image
        output_filename = f"{face_id}.jpg"
        output_path = os.path.join(processed_images_dir, output_filename)
        final_face_bgr = cv2.cvtColor(final_face_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, final_face_bgr)
        logging.info(f"Processed and saved face {output_filename}")

        # Calculate actual expansion
        actual_expansion = {
            'x_min_crop': x_min_crop,
            'y_min_crop': y_min_crop,
            'x_max_crop': x_max_crop,
            'y_max_crop': y_max_crop
        }

        # Create Face instance
        face = Face(
            id=face_id,
            photo_id=photo.id,
            image_path=output_path,
            bbox={
                'x_min': x_min_crop,
                'y_min': y_min_crop,
                'x_max': x_max_crop,
                'y_max': y_max_crop
            },
            actual_expansion=actual_expansion,
            yaw=yaw,
            pitch=pitch,
            roll=roll
        )

        faces.append(face)

    return faces
