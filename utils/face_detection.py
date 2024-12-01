import os
import cv2
import mediapipe as mp
import logging
from config import IMAGE_SIZE, DESIRED_FACE_SIZE_RATIO, PROCESSED_IMAGES_DIR
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
        expansion_ratio = 0.5  # 10% expansion
        x_min = max(int(x_min - bbox_width * expansion_ratio / 2), 0)
        y_min = max(int(y_min - bbox_height * expansion_ratio / 2), 0)
        x_max = min(int(x_min + bbox_width * (1 + expansion_ratio)), width)
        y_max = min(int(y_min + bbox_height * (1 + expansion_ratio)), height)

        # Crop the face region
        face_region = image_rgb[y_min:y_max, x_min:x_max]

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
        rotation_matrix = cv2.getRotationMatrix2D((x_min + eye_center[0], y_min + eye_center[1]), angle_degrees, 1)
        rotated_image = cv2.warpAffine(image_rgb, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)

        # Update landmarks after rotation
        ones = np.ones(shape=(len(landmarks_array), 1))
        landmarks_homogenous = np.hstack([landmarks_array + [x_min, y_min], ones])
        rotated_landmarks = rotation_matrix.dot(landmarks_homogenous.T).T

        # Recalculate key points after rotation
        left_eye_rotated = rotated_landmarks[33]
        right_eye_rotated = rotated_landmarks[263]
        eye_center_rotated = (left_eye_rotated + right_eye_rotated) / 2
        nose_tip_rotated = rotated_landmarks[1]
        mouth_center_rotated = (rotated_landmarks[13] + rotated_landmarks[14]) / 2

        # Define desired face size based on inter-eye distance
        eye_distance = np.linalg.norm(left_eye_rotated - right_eye_rotated)
        desired_face_width = eye_distance * DESIRED_FACE_SIZE_RATIO

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
        if crop_width != crop_height:
            # Adjust the smaller dimension
            if crop_width > crop_height:
                diff = crop_width - crop_height
                y_max_crop = min(y_max_crop + diff // 2, height)
                y_min_crop = max(y_min_crop - diff // 2, 0)
            else:
                diff = crop_height - crop_width
                x_max_crop = min(x_max_crop + diff // 2, width)
                x_min_crop = max(x_min_crop - diff // 2, 0)

        # Crop the aligned face
        cropped_aligned_face = rotated_image[y_min_crop:y_max_crop, x_min_crop:x_max_crop]

        # Resize to IMAGE_SIZE
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
            actual_expansion=actual_expansion
        )

        faces.append(face)

    return faces
