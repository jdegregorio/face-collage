import os
import cv2
import mediapipe as mp
from PIL import Image
import logging
from config import MIN_FACE_SIZE, IMAGE_SIZE

mp_face_detection = mp.solutions.face_detection

def process_single_image(image_path, filename, processed_images_dir):
    image = cv2.imread(image_path)
    if image is None:
        logging.warning(f"Image {filename} is unreadable.")
        return
    height, width, _ = image.shape

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.detections:
            logging.debug(f"No faces detected in {filename}.")
            return  # No faces detected

        # Process the first face detected
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        x_min = int(bbox.xmin * width)
        y_min = int(bbox.ymin * height)
        bbox_width = int(bbox.width * width)
        bbox_height = int(bbox.height * height)

        # Expand the bounding box slightly
        x_min = max(x_min - bbox_width // 10, 0)
        y_min = max(y_min - bbox_height // 10, 0)
        x_max = min(x_min + bbox_width + bbox_width // 5, width)
        y_max = min(y_min + bbox_height + bbox_height // 5, height)

        face_image = image[y_min:y_max, x_min:x_max]

        # Check if face size meets minimum criteria
        face_height, face_width = face_image.shape[:2]
        if face_height < MIN_FACE_SIZE or face_width < MIN_FACE_SIZE:
            logging.debug(f"Face in {filename} is too small.")
            return

        # Resize face image
        pil_image = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
        pil_image = pil_image.resize((IMAGE_SIZE, IMAGE_SIZE))

        # Save processed image
        output_filename = os.path.splitext(filename)[0].replace('.', '_') + '.jpg'
        output_path = os.path.join(processed_images_dir, output_filename)
        pil_image.save(output_path, format='JPEG', quality=95)
        logging.info(f"Processed and saved {output_filename}")

def process_images_batch(images, processed_images_dir):
    logging.info("Processing images...")
    for temp_image_path, filename in images:
        process_single_image(temp_image_path, filename, processed_images_dir)
