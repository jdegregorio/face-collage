import os
import cv2
import mediapipe as mp
import logging
from config import IMAGE_SIZE, DESIRED_EXPANSION, PROCESSED_IMAGES_DIR
from utils.face import Face
import uuid
import numpy as np

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

    # Face detection
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.detections:
        logging.debug(f"No faces detected in {photo.filename}.")
        return None  # No faces detected

    faces = []
    for idx, detection in enumerate(results.detections):
        bbox = detection.location_data.relative_bounding_box
        x_min = int(bbox.xmin * width)
        y_min = int(bbox.ymin * height)
        bbox_width = int(bbox.width * width)
        bbox_height = int(bbox.height * height)

        # Expand bounding box dynamically
        desired_expansion = DESIRED_EXPANSION
        x_center = x_min + bbox_width / 2
        y_center = y_min + bbox_height / 2

        max_expansion = min(
            x_center / bbox_width,
            (width - x_center) / bbox_width,
            y_center / bbox_height,
            (height - y_center) / bbox_height
        )

        actual_expansion = min(desired_expansion, max_expansion)

        # Calculate new bounding box coordinates
        new_bbox_width = bbox_width * (1 + actual_expansion)
        new_bbox_height = bbox_height * (1 + actual_expansion)
        x_min_new = int(x_center - new_bbox_width / 2)
        y_min_new = int(y_center - new_bbox_height / 2)
        x_max_new = int(x_center + new_bbox_width / 2)
        y_max_new = int(y_center + new_bbox_height / 2)

        # Ensure bounding box is within image bounds
        x_min_new = max(x_min_new, 0)
        y_min_new = max(y_min_new, 0)
        x_max_new = min(x_max_new, width)
        y_max_new = min(y_max_new, height)

        # Crop the face
        face_image = image[y_min_new:y_max_new, x_min_new:x_max_new].copy()

        # Facial landmarks for alignment and scaling
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
            face_results = face_mesh.process(face_rgb)

        if not face_results.multi_face_landmarks:
            logging.debug(f"No face landmarks found in face {idx} of {photo.filename}.")
            continue  # Skip this face

        face_landmarks = face_results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark

        # Get landmark coordinates
        left_eye = np.array([landmarks[33].x, landmarks[33].y])
        right_eye = np.array([landmarks[263].x, landmarks[263].y])
        nose_tip = np.array([landmarks[1].x, landmarks[1].y])
        mouth_center = np.array([(landmarks[13].x + landmarks[14].x) / 2, (landmarks[13].y + landmarks[14].y) / 2])

        # Calculate rotation angle
        eye_delta = right_eye - left_eye
        angle = np.arctan2(eye_delta[1], eye_delta[0])
        angle_degrees = np.degrees(angle)

        # Rotate the face image
        face_height, face_width = face_image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((face_width / 2, face_height / 2), angle_degrees, 1)
        rotated_face = cv2.warpAffine(face_image, rotation_matrix, (face_width, face_height), flags=cv2.INTER_LINEAR)

        # Update landmarks after rotation
        landmarks_array = np.array([[lm.x * face_width, lm.y * face_height] for lm in landmarks])
        ones = np.ones(shape=(len(landmarks_array), 1))
        landmarks_homogenous = np.hstack([landmarks_array, ones])
        rotated_landmarks = rotation_matrix.dot(landmarks_homogenous.T).T

        # Calculate center point (e.g., midpoint between eyes and mouth)
        eyes_center = (rotated_landmarks[33] + rotated_landmarks[263]) / 2
        face_center = (eyes_center + rotated_landmarks[1] + rotated_landmarks[13] + rotated_landmarks[14]) / 4

        # Define desired face size based on inter-eye distance
        eye_distance = np.linalg.norm(rotated_landmarks[33] - rotated_landmarks[263])
        desired_face_width = eye_distance * 4  # Adjust this factor as needed

        # Ensure the crop box is within image bounds
        half_size = desired_face_width / 2
        x_min_crop = int(face_center[0] - half_size)
        y_min_crop = int(face_center[1] - half_size)
        x_max_crop = int(face_center[0] + half_size)
        y_max_crop = int(face_center[1] + half_size)

        x_min_crop = max(x_min_crop, 0)
        y_min_crop = max(y_min_crop, 0)
        x_max_crop = min(x_max_crop, face_width)
        y_max_crop = min(y_max_crop, face_height)

        cropped_aligned_face = rotated_face[y_min_crop:y_max_crop, x_min_crop:x_max_crop]

        # Resize to IMAGE_SIZE
        final_face_image = cv2.resize(cropped_aligned_face, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

        # Generate unique ID for face
        face_id = str(uuid.uuid4())

        # Save processed face image
        output_filename = f"{face_id}.jpg"
        output_path = os.path.join(processed_images_dir, output_filename)
        cv2.imwrite(output_path, final_face_image)
        logging.info(f"Processed and saved face {output_filename}")

        # Create Face instance
        face = Face(
            id=face_id,
            photo_id=photo.id,
            image_path=output_path,
            bbox={
                'x_min': x_min_new,
                'y_min': y_min_new,
                'x_max': x_max_new,
                'y_max': y_max_new
            },
            actual_expansion=actual_expansion
        )

        faces.append(face)

    return faces
