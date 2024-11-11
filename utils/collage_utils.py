import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from datetime import datetime
import logging
from contextlib import contextmanager
import sys

from config import COLLAGE_WIDTH, COLLAGE_HEIGHT, IMAGE_SIZE, MAX_YAW, MAX_PITCH

@contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

def extract_timestamp(photo):
    if photo.creationTime:
        try:
            timestamp = datetime.strptime(photo.creationTime, '%Y-%m-%dT%H:%M:%SZ')
            return timestamp
        except Exception as e:
            logging.warning(f"Failed to parse creationTime for {photo.filename}: {e}")
    # Fallback to file modification time
    if photo.processed_image_path and os.path.exists(photo.processed_image_path):
        timestamp = datetime.fromtimestamp(os.path.getmtime(photo.processed_image_path))
        return timestamp
    else:
        return datetime.now()

def generate_collage(photos, output_path):
    logging.info("Generating the collage...")

    # Collect processed images
    valid_photos = [photo for photo in photos if photo.include_in_collage and photo.head_pose_estimation_status == 'success' and photo.processed_image_path]
    if not valid_photos:
        logging.error("No processed images with head pose estimation found.")
        print("No images available for collage generation. Please check your filters and included photos.")
        return

    images_data = []
    with suppress_stderr():
        for photo in tqdm(valid_photos, desc='Collecting image data'):
            img_path = photo.processed_image_path
            timestamp = extract_timestamp(photo)
            yaw = photo.yaw
            pitch = photo.pitch
            if yaw is None or pitch is None:
                logging.debug(f"Skipping {photo.filename} due to missing pose data.")
                continue
            if abs(yaw) > MAX_YAW or abs(pitch) > MAX_PITCH:
                logging.debug(f"Skipping {photo.filename} due to pose angles exceeding limits.")
                continue
            images_data.append({
                'image_path': img_path,
                'timestamp': timestamp,
                'yaw': yaw,
                'pitch': pitch
            })

    if not images_data:
        logging.error("No images meet the criteria for collage generation.")
        print("No images meet the criteria for collage generation.")
        return

    # Normalize yaw and timestamp
    yaws = [data['yaw'] for data in images_data]
    times = [data['timestamp'].timestamp() for data in images_data]
    min_yaw, max_yaw = min(yaws), max(yaws)
    min_time, max_time = min(times), max(times)

    # Assign grid positions
    grid_width = COLLAGE_WIDTH // IMAGE_SIZE
    grid_height = COLLAGE_HEIGHT // IMAGE_SIZE
    grid = [[None for _ in range(grid_width)] for _ in range(grid_height)]

    logging.info(f"Collage grid size: {grid_width} x {grid_height}")

    with suppress_stderr():
        for data in tqdm(images_data, desc='Placing images'):
            norm_yaw = (data['yaw'] - min_yaw) / (max_yaw - min_yaw) if max_yaw > min_yaw else 0.5
            norm_time = (data['timestamp'].timestamp() - min_time) / (max_time - min_time) if max_time > min_time else 0.5

            x = int(norm_yaw * (grid_width - 1))
            y = int(norm_time * (grid_height - 1))

            if grid[y][x] is None:
                grid[y][x] = data['image_path']
            else:
                # Find nearest empty spot
                found = False
                for offset in range(1, max(grid_width, grid_height)):
                    for dx in range(-offset, offset + 1):
                        for dy in range(-offset, offset + 1):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < grid_width and 0 <= ny < grid_height and grid[ny][nx] is None:
                                grid[ny][nx] = data['image_path']
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break

    # Create a blank canvas
    with suppress_stderr():
        collage_image = Image.new('RGB', (grid_width * IMAGE_SIZE, grid_height * IMAGE_SIZE))

    # Paste images onto the canvas
    with suppress_stderr():
        for y in tqdm(range(grid_height), desc='Creating collage'):
            for x in range(grid_width):
                img_path = grid[y][x]
                if img_path is not None:
                    img = Image.open(img_path)
                    collage_image.paste(img, (x * IMAGE_SIZE, y * IMAGE_SIZE))
                else:
                    # Optionally fill with a placeholder
                    pass

    # Save the collage
    with suppress_stderr():
        collage_image.save(output_path)
    logging.info(f"Collage saved to {output_path}")
    print(f"Collage saved to {output_path}")
