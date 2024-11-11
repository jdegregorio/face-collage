import os
import logging
from simple_term_menu import TerminalMenu
import pandas as pd
from tqdm import tqdm
import sys
import json
from datetime import datetime, timedelta, date

from utils.google_photos_api import (
    list_albums,
    get_album_photos,
    download_photo
)
from utils.face_detection import process_single_image
from utils.collage_utils import generate_collage
from utils.progress_tracker import ProgressTracker
from utils.photo import Photo
from utils.filtering import (
    exclude_failed_processing_photos,
    filter_photos_by_features,
    filter_photos_by_date,  # Updated function
    update_status_based_on_file_existence,
    reset_filters
)
from config import *

# Suppress TensorFlow and Mediapipe logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

# Suppress absl logging from TensorFlow and Mediapipe
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except ImportError:
    pass  # absl not installed

def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(LOG_DIR, 'app.log'),
        level=getattr(logging, LOG_LEVEL),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def create_directories():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_IMAGES_DIR, exist_ok=True)
    os.makedirs(ORIGINAL_IMAGES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(EXCLUDED_IMAGES_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs('credentials', exist_ok=True)

def main_menu():
    options = [
        "1. Index Photos in Album",
        "2. Download, Crop & Resize Photos",
        "3. Perform Head Pose and Facial Features Estimation",
        "4. Filter and Manage Photos",
        "5. Generate Collage",
        "6. View Progress",
        "7. Reset Project",
        "8. Exit"
    ]
    terminal_menu = TerminalMenu(options, title="Image Collage Project - Main Menu")
    menu_entry_index = terminal_menu.show()
    return menu_entry_index

def select_album():
    albums = list_albums()
    if not albums:
        print("No albums found.")
        return None

    album_titles = [f"{i+1}. {album.get('title', 'Untitled')} (ID: {album['id']})" for i, album in enumerate(albums)]
    terminal_menu = TerminalMenu(album_titles, title="Select an Album")
    menu_entry_index = terminal_menu.show()
    if menu_entry_index is None:
        return None

    selected_album = albums[menu_entry_index]
    return selected_album

def parse_creation_time(creation_time_str):
    # Try parsing with milliseconds
    try:
        timestamp = datetime.strptime(creation_time_str, '%Y-%m-%dT%H:%M:%S.%fZ')
        return timestamp
    except ValueError:
        pass  # Try next format

    # Try parsing without milliseconds
    try:
        timestamp = datetime.strptime(creation_time_str, '%Y-%m-%dT%H:%M:%SZ')
        return timestamp
    except ValueError:
        pass  # All parsing attempts failed

    logging.warning(f"Failed to parse creationTime: {creation_time_str}")
    return None

def create_photo_index(index_file, photos_file, album_id):
    """
    Create an index of all photos in the specified album.
    """
    media_items = get_album_photos(album_id)
    if not media_items:
        logging.error("No media items found in the album.")
        return

    photos = []
    for item in media_items:
        if 'image' in item['mimeType']:
            # Parse creationTime
            creationTime = item.get('mediaMetadata', {}).get('creationTime')
            timestamp = None
            year = month = day = weekday = None
            date_floor_day = date_floor_week = date_floor_month = date_floor_year = None
            if creationTime:
                timestamp = parse_creation_time(creationTime)
                if timestamp:
                    year = timestamp.year
                    month = timestamp.month
                    day = timestamp.day
                    weekday = timestamp.weekday()  # Monday is 0
                    date_floor_day = timestamp.date()  # Floor to day
                    date_floor_week = date_floor_day - timedelta(days=timestamp.weekday())
                    date_floor_month = date(timestamp.year, timestamp.month, 1)
                    date_floor_year = date(timestamp.year, 1, 1)
                else:
                    logging.warning(f"Timestamp is None for {item.get('filename')} after parsing creationTime: {creationTime}")

            # Create Photo instance
            photo = Photo(
                id=item.get('id'),
                filename=item.get('filename'),
                description=item.get('description'),
                mimeType=item.get('mimeType'),
                productUrl=item.get('productUrl'),
                baseUrl=item.get('baseUrl'),
                creationTime=creationTime,
                width=int(item.get('mediaMetadata', {}).get('width', 0)) if item.get('mediaMetadata', {}).get('width') else None,
                height=int(item.get('mediaMetadata', {}).get('height', 0)) if item.get('mediaMetadata', {}).get('height') else None,
                photo_cameraMake=item.get('mediaMetadata', {}).get('photo', {}).get('cameraMake'),
                photo_cameraModel=item.get('mediaMetadata', {}).get('photo', {}).get('cameraModel'),
                photo_focalLength=float(item.get('mediaMetadata', {}).get('photo', {}).get('focalLength', 0)) if item.get('mediaMetadata', {}).get('photo', {}).get('focalLength') else None,
                photo_apertureFNumber=float(item.get('mediaMetadata', {}).get('photo', {}).get('apertureFNumber', 0)) if item.get('mediaMetadata', {}).get('photo', {}).get('apertureFNumber') else None,
                photo_isoEquivalent=int(item.get('mediaMetadata', {}).get('photo', {}).get('isoEquivalent', 0)) if item.get('mediaMetadata', {}).get('photo', {}).get('isoEquivalent') else None,
                timestamp=timestamp,
                year=year,
                month=month,
                day=day,
                weekday=weekday,
                date_floor_day=date_floor_day,
                date_floor_week=date_floor_week,
                date_floor_month=date_floor_month,
                date_floor_year=date_floor_year,
            )
            photos.append(photo)
        else:
            logging.info(f"Skipping non-image media item: {item.get('filename')}")

    # Save photos to JSON
    save_photos(photos, photos_file)
    logging.info(f"Photo data saved to {photos_file}")

    # Save index to CSV for reporting
    update_index_csv(photos, index_file)
    print(f"Indexed {len(photos)} photos from the album.")

def save_photos(photos, file_path):
    with open(file_path, 'w') as f:
        json.dump([photo.to_dict() for photo in photos], f, indent=4)

def load_photos(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        photos = [Photo.from_dict(photo_data) for photo_data in data]
        return photos
    else:
        return []

def update_index_csv(photos, index_file):
    df = pd.DataFrame([photo.to_dict() for photo in photos])
    df.to_csv(index_file, index=False)
    logging.info(f"Photo index updated and saved to {index_file}")

def download_crop_resize_photos(photos):
    """
    Download, crop, and resize photos one by one.
    """
    total_photos = len(photos)
    logging.info(f"Total photos to process: {total_photos}")

    photos_to_process = [photo for photo in photos if photo.download_status != 'success' or photo.face_detection_status != 'success']

    if not photos_to_process:
        print("All photos have been processed.")
        return

    for photo in tqdm(photos_to_process, desc="Processing photos", unit="photo"):
        try:
            # Download photo
            if photo.download_status != 'success':
                temp_image_path = download_photo(photo.id, photo.filename, ORIGINAL_IMAGES_DIR)
                if temp_image_path:
                    photo.original_image_path = temp_image_path
                    photo.download_status = 'success'
                    photo.download_error = ''
                else:
                    photo.download_status = 'failed'
                    photo.download_error = 'Download failed'
                    logging.warning(f"Failed to download photo {photo.filename}")
                    # Proceed to delete the original image even if download fails
                    if DELETE_ORIGINAL_AFTER_PROCESSING and os.path.exists(photo.original_image_path):
                        os.remove(photo.original_image_path)
                        photo.original_image_path = ''
                    # Save progress and continue to next photo
                    save_photos(photos, PHOTOS_FILE)
                    update_index_csv(photos, INDEX_FILE)
                    continue  # Skip to next photo

            # Process image
            if photo.face_detection_status != 'success':
                result = process_single_image(photo)
                if result:
                    photo.processed_image_path = result
                    photo.face_detection_status = 'success'
                    photo.face_detection_error = ''
                else:
                    photo.face_detection_status = 'failed'
                    photo.face_detection_error = 'Face detection failed'
                    logging.warning(f"Failed to process photo {photo.filename}")
                    # Proceed to delete the original image even if face detection fails
                    if DELETE_ORIGINAL_AFTER_PROCESSING and os.path.exists(photo.original_image_path):
                        os.remove(photo.original_image_path)
                        photo.original_image_path = ''
                    # Save progress and continue to next photo
                    save_photos(photos, PHOTOS_FILE)
                    update_index_csv(photos, INDEX_FILE)
                    continue  # Skip to next photo

            # Delete original image if configured
            if DELETE_ORIGINAL_AFTER_PROCESSING and os.path.exists(photo.original_image_path):
                os.remove(photo.original_image_path)
                photo.original_image_path = ''

            # Save progress after each photo
            save_photos(photos, PHOTOS_FILE)
            update_index_csv(photos, INDEX_FILE)

        except Exception as e:
            logging.error(f"Unexpected error processing photo {photo.filename}: {e}")
            photo.download_status = 'failed'
            photo.face_detection_status = 'failed'
            photo.download_error = 'Unexpected error'
            photo.face_detection_error = 'Unexpected error'
            # Proceed to delete the original image even if an unexpected error occurs
            if DELETE_ORIGINAL_AFTER_PROCESSING and os.path.exists(photo.original_image_path):
                os.remove(photo.original_image_path)
                photo.original_image_path = ''
            # Save progress and continue to next photo
            save_photos(photos, PHOTOS_FILE)
            update_index_csv(photos, INDEX_FILE)
            continue  # Skip to next photo

def perform_head_pose_and_facial_features_estimation(photos):
    """
    Perform head pose estimation and extract facial features on processed photos.
    """
    total_photos = len(photos)
    logging.info(f"Total photos to process: {total_photos}")

    photos_to_process = [photo for photo in photos if photo.face_detection_status == 'success' and photo.head_pose_estimation_status != 'success']

    if not photos_to_process:
        print("All photos have head pose estimation completed.")
        return

    from utils.head_pose_and_facial_features import estimate_head_pose_and_facial_features

    for photo in tqdm(photos_to_process, desc="Estimating head poses and extracting facial features", unit="photo"):
        try:
            if photo.head_pose_estimation_status != 'success':
                result = estimate_head_pose_and_facial_features(photo.processed_image_path)
                if result:
                    yaw, pitch, roll, facial_features = result
                    photo.yaw = yaw
                    photo.pitch = pitch
                    photo.roll = roll
                    photo.head_pose_estimation_status = 'success'
                    photo.head_pose_estimation_error = ''
                    # Update facial features
                    photo.left_eye_openness = facial_features['left_eye_openness']
                    photo.right_eye_openness = facial_features['right_eye_openness']
                    photo.avg_eye_openness = facial_features['avg_eye_openness']
                    photo.mouth_openness = facial_features['mouth_openness']
                    photo.facial_features_status = 'success'
                else:
                    photo.head_pose_estimation_status = 'failed'
                    photo.head_pose_estimation_error = 'Head pose estimation failed'
                    photo.facial_features_status = 'failed'
                    logging.warning(f"Failed to estimate head pose for photo {photo.filename}")
                # Save progress after each photo
                save_photos(photos, PHOTOS_FILE)
                update_index_csv(photos, INDEX_FILE)

        except Exception as e:
            logging.error(f"Unexpected error processing photo {photo.filename}: {e}")
            photo.head_pose_estimation_status = 'failed'
            photo.head_pose_estimation_error = 'Unexpected error'
            photo.facial_features_status = 'failed'
            # Save progress and continue to next photo
            save_photos(photos, PHOTOS_FILE)
            update_index_csv(photos, INDEX_FILE)
            continue  # Skip to next photo

def reset_project(tracker):
    """
    Reset the project's state.
    """
    confirmation = input("Are you sure you want to reset the project? This will delete all progress and processed data. (yes/no): ")
    if confirmation.lower() == 'yes':
        # Reset tracker
        tracker.reset_progress()

        # Delete processed images
        if os.path.exists(PROCESSED_IMAGES_DIR):
            for filename in os.listdir(PROCESSED_IMAGES_DIR):
                file_path = os.path.join(PROCESSED_IMAGES_DIR, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        # Delete original images
        if os.path.exists(ORIGINAL_IMAGES_DIR):
            for filename in os.listdir(ORIGINAL_IMAGES_DIR):
                file_path = os.path.join(ORIGINAL_IMAGES_DIR, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        # Delete excluded images
        if os.path.exists(EXCLUDED_IMAGES_DIR):
            for filename in os.listdir(EXCLUDED_IMAGES_DIR):
                file_path = os.path.join(EXCLUDED_IMAGES_DIR, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        # Delete index file
        if os.path.exists(INDEX_FILE):
            os.remove(INDEX_FILE)

        # Delete photos data file
        if os.path.exists(PHOTOS_FILE):
            os.remove(PHOTOS_FILE)

        # Delete progress file
        if os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)

        print("Project has been reset.")
        logging.info("Project has been reset.")
    else:
        print("Project reset canceled.")

def filter_and_manage_photos(photos):
    while True:
        options = [
            "1. Exclude Photos with Failed Processing",
            "2. Filter Photos by Head Pose and Facial Features",
            "3. Filter Photos by Date Range",
            "4. Update Status Based on File Existence",
            "5. Reset All Filters",
            "6. Return to Main Menu"
        ]
        terminal_menu = TerminalMenu(options, title="Filter and Manage Photos")
        menu_entry_index = terminal_menu.show()

        if menu_entry_index == 0:
            # Exclude Photos with Failed Processing
            exclude_failed_processing_photos(photos)
            save_photos(photos, PHOTOS_FILE)
            update_index_csv(photos, INDEX_FILE)
        elif menu_entry_index == 1:
            # Filter Photos by Head Pose and Facial Features
            filter_photos_by_features(photos)
            save_photos(photos, PHOTOS_FILE)
            update_index_csv(photos, INDEX_FILE)
        elif menu_entry_index == 2:
            # Filter Photos by Date Range
            filter_photos_by_date(photos)
            save_photos(photos, PHOTOS_FILE)
            update_index_csv(photos, INDEX_FILE)
        elif menu_entry_index == 3:
            # Update Status Based on File Existence
            update_status_based_on_file_existence(photos)
            save_photos(photos, PHOTOS_FILE)
            update_index_csv(photos, INDEX_FILE)
        elif menu_entry_index == 4:
            # Reset All Filters
            reset_filters(photos)
            save_photos(photos, PHOTOS_FILE)
            update_index_csv(photos, INDEX_FILE)
        elif menu_entry_index == 5:
            # Return to Main Menu
            break
        else:
            print("Invalid selection. Please try again.")

def main():
    setup_logging()
    create_directories()
    tracker = ProgressTracker(PROGRESS_FILE)

    while True:
        selection = main_menu()
        if selection == 0:
            # Index Photos in Album
            selected_album = select_album()
            if selected_album:
                album_id = selected_album['id']
                create_photo_index(INDEX_FILE, PHOTOS_FILE, album_id)
                tracker.update_stage('Indexing Completed')
            else:
                print("No album selected.")

        elif selection == 1:
            # Download, Crop & Resize Photos
            photos = load_photos(PHOTOS_FILE)
            if not photos:
                print("No photos to process. Please index an album first.")
                continue
            download_crop_resize_photos(photos)
            tracker.update_stage('Download and Processing Completed')

        elif selection == 2:
            # Perform Head Pose and Facial Features Estimation
            photos = load_photos(PHOTOS_FILE)
            if not photos:
                print("No photos to process. Please index an album first.")
                continue
            perform_head_pose_and_facial_features_estimation(photos)
            tracker.update_stage('Head Pose and Facial Features Estimation Completed')

        elif selection == 3:
            # Filter and Manage Photos
            photos = load_photos(PHOTOS_FILE)
            if not photos:
                print("No photos to process. Please index an album first.")
                continue
            filter_and_manage_photos(photos)
            tracker.update_stage('Filtering and Management Completed')

        elif selection == 4:
            # Generate Collage
            photos = load_photos(PHOTOS_FILE)
            if not photos:
                print("No photos to process. Please index an album first.")
                continue
            generate_collage(photos, COLLAGE_OUTPUT_PATH)
            tracker.update_stage('Collage Generated')

        elif selection == 5:
            # View Progress
            if os.path.exists(PHOTOS_FILE):
                photos = load_photos(PHOTOS_FILE)
            else:
                photos = []
            tracker.display_progress(photos)
            input("\nPress Enter to return to the main menu...")

        elif selection == 6:
            # Reset Project
            reset_project(tracker)

        elif selection == 7:
            # Exit
            print("Exiting the application. Goodbye!")
            break

        else:
            print("Invalid selection. Please try again.")

if __name__ == '__main__':
    main()
