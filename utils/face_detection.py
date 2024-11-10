import os
import cv2
import mediapipe as mp
from PIL import Image
import logging
from tqdm import tqdm
from config import IMAGE_SIZE, INITIAL_BBOX_EXPANSION
import sys

mp_face_detection = mp.solutions.face_detection

def process_single_image(args):
    temp_image_path, filename, processed_images_dir = args
    image = cv2.imread(temp_image_path)
    if image is None:
        logging.warning(f"Image {filename} is unreadable.")
        return
    height, width, _ = image.shape

    # Step 1: Initial face detection to get a rough bounding box
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

    # Expand the bounding box based on INITIAL_BBOX_EXPANSION
    expansion = INITIAL_BBOX_EXPANSION
    x_min = max(int(x_min - bbox_width * expansion / 2), 0)
    y_min = max(int(y_min - bbox_height * expansion / 2), 0)
    x_max = min(int(x_min + bbox_width * (1 + expansion)), width)
    y_max = min(int(y_min + bbox_height * (1 + expansion)), height)

    face_image = image[y_min:y_max, x_min:x_max].copy()

    # Ensure the face is centered in the image
    face_height, face_width = face_image.shape[:2]
    desired_size = max(face_width, face_height)
    delta_w = desired_size - face_width
    delta_h = desired_size - face_height
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [0, 0, 0]
    face_image = cv2.copyMakeBorder(face_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

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
    args_list = [(temp_image_path, filename, processed_images_dir) for temp_image_path, filename in images]
    for _ in tqdm(map(process_single_image, args_list), total=len(args_list), desc='Processing images', file=sys.stdout):
        pass
