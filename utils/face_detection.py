import os
import cv2
import mediapipe as mp
from PIL import Image
import logging
from tqdm import tqdm
from config import MIN_FACE_SIZE, IMAGE_SIZE, INITIAL_BBOX_EXPANSION, LANDMARK_BBOX_MARGIN
from contextlib import contextmanager
import sys

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

@contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

def process_single_image(args):
    temp_image_path, filename, processed_images_dir = args
    image = cv2.imread(temp_image_path)
    if image is None:
        logging.warning(f"Image {filename} is unreadable.")
        return
    height, width, _ = image.shape

    # Step 1: Initial face detection to get a rough bounding box
    with suppress_stderr():
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
    final_face_image = face_image

    # # Step 2: Use face mesh to get facial landmarks for precise cropping
    # with suppress_stderr():
    #     with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
    #         face_results = face_mesh.process(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))

    # if not face_results.multi_face_landmarks:
    #     logging.debug(f"No face landmarks found in {filename}.")
    #     return

    # face_landmarks = face_results.multi_face_landmarks[0]

    # # Get the bounding rectangle around the landmarks
    # x_coords = [landmark.x for landmark in face_landmarks.landmark]
    # y_coords = [landmark.y for landmark in face_landmarks.landmark]
    # xmin = min(x_coords)
    # xmax = max(x_coords)
    # ymin = min(y_coords)
    # ymax = max(y_coords)

    # # Convert normalized coordinates to pixel values
    # face_height_img, face_width_img = face_image.shape[:2]
    # xmin = int(xmin * face_width_img)
    # xmax = int(xmax * face_width_img)
    # ymin = int(ymin * face_height_img)
    # ymax = int(ymax * face_height_img)

    # # Apply margin around the landmarks
    # margin = LANDMARK_BBOX_MARGIN
    # x_center = (xmin + xmax) / 2
    # y_center = (ymin + ymax) / 2
    # bbox_width_final = xmax - xmin
    # bbox_height_final = ymax - ymin
    # bbox_size = max(bbox_width_final, bbox_height_final) * (1 + margin)

    # xmin = int(x_center - bbox_size / 2)
    # xmax = int(x_center + bbox_size / 2)
    # ymin = int(y_center - bbox_size / 2)
    # ymax = int(y_center + bbox_size / 2)

    # # Ensure coordinates are within image boundaries
    # xmin = max(xmin, 0)
    # ymin = max(ymin, 0)
    # xmax = min(xmax, face_width_img)
    # ymax = min(ymax, face_height_img)

    # # Final cropped face image
    # final_face_image = face_image[ymin:ymax, xmin:xmax]

    # Check if face size meets minimum criteria
    final_face_height, final_face_width = final_face_image.shape[:2]
    if final_face_height < MIN_FACE_SIZE or final_face_width < MIN_FACE_SIZE:
        logging.debug(f"Face in {filename} is too small after cropping.")
        return

    # Resize face image
    pil_image = Image.fromarray(cv2.cvtColor(final_face_image, cv2.COLOR_BGR2RGB))
    pil_image = pil_image.resize((IMAGE_SIZE, IMAGE_SIZE))

    # Save processed image
    output_filename = os.path.splitext(filename)[0].replace('.', '_') + '.jpg'
    output_path = os.path.join(processed_images_dir, output_filename)
    pil_image.save(output_path, format='JPEG', quality=95)
    logging.info(f"Processed and saved {output_filename}")

def process_images_batch(images, processed_images_dir):
    logging.info("Processing images...")
    args_list = [(temp_image_path, filename, processed_images_dir) for temp_image_path, filename in images]
    for _ in tqdm(map(process_single_image, args_list), total=len(args_list), desc='Processing images'):
        pass
