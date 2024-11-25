import os
import cv2
import mediapipe as mp
import logging
from config import IMAGE_SIZE, INITIAL_BBOX_EXPANSION, PROCESSED_IMAGES_DIR
from utils.face import Face
import uuid

mp_face_detection = mp.solutions.face_detection

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

        # Expand bounding box
        expansion_ratio = INITIAL_BBOX_EXPANSION
        x_min_new = max(int(x_min - bbox_width * expansion_ratio / 2), 0)
        y_min_new = max(int(y_min - bbox_height * expansion_ratio / 2), 0)
        x_max_new = min(int(x_min + bbox_width * (1 + expansion_ratio / 2)), width)
        y_max_new = min(int(y_min + bbox_height * (1 + expansion_ratio / 2)), height)

        # Crop the face
        face_image = image[y_min_new:y_max_new, x_min_new:x_max_new].copy()

        # Ensure the face image is square by padding
        face_height, face_width = face_image.shape[:2]
        max_side = max(face_width, face_height)
        padded_face = cv2.copyMakeBorder(
            face_image,
            top=(max_side - face_height) // 2,
            bottom=(max_side - face_height + 1) // 2,
            left=(max_side - face_width) // 2,
            right=(max_side - face_width + 1) // 2,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )

        # Resize face image to desired IMAGE_SIZE
        face_image_resized = cv2.resize(padded_face, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

        # Convert to RGB since face_recognition expects RGB images
        face_image_resized = cv2.cvtColor(face_image_resized, cv2.COLOR_BGR2RGB)

        # Generate unique ID for face
        face_id = str(uuid.uuid4())

        # Save processed face image
        output_filename = f"{face_id}.jpg"
        output_path = os.path.join(processed_images_dir, output_filename)
        cv2.imwrite(output_path, cv2.cvtColor(face_image_resized, cv2.COLOR_RGB2BGR))
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
            actual_expansion=expansion_ratio
        )

        faces.append(face)

    return faces
