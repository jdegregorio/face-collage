import os
import cv2
import mediapipe as mp
import logging
from config import IMAGE_SIZE, INITIAL_BBOX_EXPANSION, PROCESSED_IMAGES_DIR

mp_face_detection = mp.solutions.face_detection

def process_single_image(photo):
    image_path = photo.original_image_path
    processed_images_dir = PROCESSED_IMAGES_DIR

    image = cv2.imread(image_path)
    if image is None:
        logging.warning(f"Image {photo.filename} is unreadable.")
        return None
    height, width, _ = image.shape

    # Step 1: Initial face detection to get a rough bounding box
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.detections:
        logging.debug(f"No faces detected in {photo.filename}.")
        return None  # No faces detected

    # Process the first face detected
    detection = results.detections[0]
    bbox = detection.location_data.relative_bounding_box
    x_min = int(bbox.xmin * width)
    y_min = int(bbox.ymin * height)
    bbox_width = int(bbox.width * width)
    bbox_height = int(bbox.height * height)

    # Desired expansion in pixels
    desired_expansion_w = int(bbox_width * INITIAL_BBOX_EXPANSION)
    desired_expansion_h = int(bbox_height * INITIAL_BBOX_EXPANSION)

    # Calculate maximum possible expansion without going outside image boundaries
    max_expand_left = x_min
    max_expand_right = width - (x_min + bbox_width)
    max_expand_up = y_min
    max_expand_down = height - (y_min + bbox_height)

    # Actual expansion to apply
    expand_left = min(desired_expansion_w // 2, max_expand_left)
    expand_right = min(desired_expansion_w - expand_left, max_expand_right)
    expand_up = min(desired_expansion_h // 2, max_expand_up)
    expand_down = min(desired_expansion_h - expand_up, max_expand_down)

    # Adjust x_min, y_min, x_max, y_max
    x_min_new = x_min - expand_left
    x_max_new = x_min + bbox_width + expand_right
    y_min_new = y_min - expand_up
    y_max_new = y_min + bbox_height + expand_down

    # Calculate actual bounding box size
    actual_bbox_width = x_max_new - x_min_new
    actual_bbox_height = y_max_new - y_min_new

    # Calculate actual expansion ratio
    expansion_w = (actual_bbox_width - bbox_width) / bbox_width
    expansion_h = (actual_bbox_height - bbox_height) / bbox_height
    actual_expansion_ratio = (expansion_w + expansion_h) / 2

    # Update photo with actual expansion applied
    photo.actual_expansion = actual_expansion_ratio

    # Crop the image
    face_image = image[y_min_new:y_max_new, x_min_new:x_max_new].copy()

    # Ensure the face image is square by further cropping if necessary
    face_height, face_width = face_image.shape[:2]
    if face_width != face_height:
        # Find the center coordinates
        center_x = face_width // 2
        center_y = face_height // 2

        # Determine the size of the square we can crop
        side_length = min(face_width, face_height)

        # Calculate the top-left corner of the square
        x1 = center_x - side_length // 2
        y1 = center_y - side_length // 2

        # Ensure the coordinates are within the image boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = x1 + side_length
        y2 = y1 + side_length

        # Crop the image to the square
        face_image = face_image[y1:y2, x1:x2]

    # Resize face image to desired IMAGE_SIZE
    face_image = cv2.resize(face_image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

    # Save processed image using photo.id
    output_filename = f"{photo.id}.jpg"
    output_path = os.path.join(processed_images_dir, output_filename)
    cv2.imwrite(output_path, face_image)
    logging.info(f"Processed and saved {output_filename}")
    return output_path
