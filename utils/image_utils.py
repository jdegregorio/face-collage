import logging
from config import MAX_PITCH, MAX_YAW

def filter_images(pose_data, max_yaw=MAX_YAW, max_pitch=MAX_PITCH):
    logging.info("Filtering images based on head pose angles...")
    filtered_data = []
    for data in pose_data:
        yaw = abs(data.get('yaw', 0))
        pitch = abs(data.get('pitch', 0))

        if yaw > max_yaw or pitch > max_pitch:
            logging.debug(f"Image {data['image_path']} filtered out: yaw {yaw}, pitch {pitch}")
            continue  # Exclude images with large head turns

        filtered_data.append(data)

    logging.info(f"{len(filtered_data)} images passed the filtering criteria.")

    return filtered_data
