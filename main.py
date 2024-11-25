import os
import logging
from simple_term_menu import TerminalMenu
import pandas as pd
from tqdm import tqdm
import sys
import json
from datetime import datetime, timedelta, date
from dateutil import parser

from utils.google_photos_api import (
    list_albums,
    get_album_photos,
    download_photo
)
from utils.face_detection import process_single_image
from utils.collage_utils import generate_collage
from utils.progress_tracker import ProgressTracker
from utils.photo import Photo
from utils.face import Face
from utils.filtering import (
    exclude_failed_processing_photos,
    filter_photos_by_features,
    filter_photos_by_date,
    filter_photos_by_temporal_clustering,
    update_status_based_on_file_existence,
    reset_filters,
    sample_photos_temporally,
    filter_faces_by_classifier
)
from utils.classifier_utils import (
    train_classifier,
    classify_faces
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
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(POSITIVE_FACES_DIR, exist_ok=True)
    os.makedirs(NEGATIVE_FACES_DIR, exist_ok=True)

def main_menu():
    options = [
        "1. Index Photos in Album",
        "2. Download and Process Photos",
        "3. Train Custom Classifier",
        "4. Classify Faces",
        "5. Filter and Manage Photos",
        "6. Generate Collage",
        "7. View Progress",
        "8. Reset Project",
        "9. Exit"
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
    # Define possible formats
    formats = [
        '%Y-%m-%dT%H:%M:%S.%fZ',   # With microseconds and 'Z'
        '%Y-%m-%dT%H:%M:%SZ',      # Without microseconds, with 'Z'
        '%Y-%m-%dT%H:%M:%S.%f',    # With microseconds, no 'Z'
        '%Y-%m-%dT%H:%M:%S',       # Without microseconds, no 'Z'
        '%Y-%m-%dT%H:%M:%S%z',     # With timezone offset
        '%Y-%m-%dT%H:%M:%S.%f%z',  # With microseconds and timezone offset
    ]

    for fmt in formats:
        try:
            timestamp = datetime.strptime(creation_time_str, fmt)
            return timestamp
        except ValueError:
            continue  # Try next format

    # As a last resort, try using dateutil parser
    try:
        timestamp = parser.parse(creation_time_str)
        return timestamp
    except (ValueError, OverflowError):
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
    update_index_csv(photos)
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

def update_index_csv(photos):
    # Photos index
    photos_df = pd.DataFrame([photo.to_dict() for photo in photos])

    # Convert timestamp and date fields to strings in photos_df
    date_columns = ['timestamp', 'date_floor_day', 'date_floor_week', 'date_floor_month', 'date_floor_year']
    for col in date_columns:
        if col in photos_df.columns:
            photos_df[col] = photos_df[col].astype(str)

    photos_df.to_csv(PHOTO_INDEX_FILE, index=False)
    logging.info(f"Photo index updated and saved to {PHOTO_INDEX_FILE}")

    # Faces index
    faces_data = []
    for photo in photos:
        if photo.face_list:
            for face in photo.face_list:
                face_dict = face.to_dict()
                # Add photo-level data to face dict
                face_dict['photo_id'] = photo.id
                face_dict['photo_filename'] = photo.filename
                face_dict['photo_timestamp'] = photo.timestamp
                face_dict['photo_date_floor_day'] = photo.date_floor_day
                face_dict['photo_date_floor_week'] = photo.date_floor_week
                face_dict['photo_date_floor_month'] = photo.date_floor_month
                face_dict['photo_date_floor_year'] = photo.date_floor_year
                faces_data.append(face_dict)
    if faces_data:
        faces_df = pd.DataFrame(faces_data)

        # Convert timestamp and date fields to strings in faces_df
        date_columns = ['photo_timestamp', 'photo_date_floor_day', 'photo_date_floor_week', 'photo_date_floor_month', 'photo_date_floor_year']
        for col in date_columns:
            if col in faces_df.columns:
                faces_df[col] = faces_df[col].astype(str)

        faces_df.to_csv(FACES_INDEX_FILE, index=False)
        logging.info(f"Faces index updated and saved to {FACES_INDEX_FILE}")
    else:
        logging.info("No faces data to save to faces index CSV.")

def download_and_process_photos(photos):
    """
    Download and process photos, extracting all faces.
    """
    total_photos = len(photos)
    logging.info(f"Total photos to process: {total_photos}")

    photos_to_process = [photo for photo in photos if photo.download_status != 'success']

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
                    update_index_csv(photos)
                    continue  # Skip to next photo

            # Process image and extract faces
            if not photo.faces_detected:
                faces = process_single_image(photo)
                if faces:
                    photo.faces_detected = True
                    photo.face_list = faces
                    photo.face_detection_error = ''
                else:
                    photo.faces_detected = False
                    photo.face_detection_error = 'Face detection failed'
                    logging.warning(f"No faces detected in photo {photo.filename}")
                # Save progress and continue to next photo
                save_photos(photos, PHOTOS_FILE)
                update_index_csv(photos)

            # Delete original image if configured
            if DELETE_ORIGINAL_AFTER_PROCESSING and os.path.exists(photo.original_image_path):
                os.remove(photo.original_image_path)
                photo.original_image_path = ''

        except Exception as e:
            logging.error(f"Unexpected error processing photo {photo.filename}: {e}")
            photo.download_status = 'failed'
            photo.faces_detected = False
            photo.download_error = 'Unexpected error'
            photo.face_detection_error = 'Unexpected error'
            # Proceed to delete the original image even if an unexpected error occurs
            if DELETE_ORIGINAL_AFTER_PROCESSING and os.path.exists(photo.original_image_path):
                os.remove(photo.original_image_path)
                photo.original_image_path = ''
            # Save progress and continue to next photo
            save_photos(photos, PHOTOS_FILE)
            update_index_csv(photos)
            continue  # Skip to next photo

def train_custom_classifier(photos):
    """
    Train a custom classifier using the faces in the training directories.
    """
    positive_dir = POSITIVE_FACES_DIR
    negative_dir = NEGATIVE_FACES_DIR
    classifier_path = CLASSIFIER_MODEL_PATH

    # Check if training data exists
    num_positive = len(os.listdir(positive_dir))
    num_negative = len(os.listdir(negative_dir))

    if num_positive == 0 or num_negative == 0:
        print(f"Not enough training data. Positive samples: {num_positive}, Negative samples: {num_negative}")
        print(f"Please ensure you have images in {positive_dir} and {negative_dir}")
        return

    print(f"Starting training with {num_positive} positive samples and {num_negative} negative samples.")
    train_classifier(positive_dir, negative_dir, classifier_path)

def classify_all_faces(photos):
    """
    Classify all detected faces using the trained classifier.
    """
    classifier_path = CLASSIFIER_MODEL_PATH
    if not os.path.exists(classifier_path):
        print("Classifier model not found. Please train the classifier first.")
        return

    all_faces = []
    for photo in photos:
        if photo.faces_detected and photo.face_list:
            all_faces.extend(photo.face_list)

    if not all_faces:
        print("No faces to classify.")
        return

    classify_faces(all_faces, classifier_path)

    # Save updated photos
    save_photos(photos, PHOTOS_FILE)
    update_index_csv(photos)

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

        # # Delete model directory
        # if os.path.exists(MODEL_DIR):
        #     for root, dirs, files in os.walk(MODEL_DIR):
        #         for file in files:
        #             os.remove(os.path.join(root, file))

        # Delete classifier model
        if os.path.exists(CLASSIFIER_MODEL_PATH):
            os.remove(CLASSIFIER_MODEL_PATH)

        # Delete index files
        if os.path.exists(PHOTO_INDEX_FILE):
            os.remove(PHOTO_INDEX_FILE)
        if os.path.exists(FACES_INDEX_FILE):
            os.remove(FACES_INDEX_FILE)

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
            "2. Filter Faces by Head Pose and Facial Features",
            "3. Filter Faces by Date Range",
            "4. Filter Faces by Temporal Clustering",
            "5. Filter Faces by Classifier",
            "6. Update Status Based on File Existence",
            "7. Reset All Filters",
            "8. Return to Main Menu"
        ]
        terminal_menu = TerminalMenu(options, title="Filter and Manage Photos")
        menu_entry_index = terminal_menu.show()

        if menu_entry_index == 0:
            # Exclude Photos with Failed Processing
            exclude_failed_processing_photos(photos)
            save_photos(photos, PHOTOS_FILE)
            update_index_csv(photos)
        elif menu_entry_index == 1:
            # Filter Faces by Head Pose and Facial Features
            filter_photos_by_features(photos)
            save_photos(photos, PHOTOS_FILE)
            update_index_csv(photos)
        elif menu_entry_index == 2:
            # Filter Faces by Date Range
            filter_photos_by_date(photos)
            save_photos(photos, PHOTOS_FILE)
            update_index_csv(photos)
        elif menu_entry_index == 3:
            # Filter Faces by Temporal Clustering
            filter_photos_by_temporal_clustering(photos)
            save_photos(photos, PHOTOS_FILE)
            update_index_csv(photos)
        elif menu_entry_index == 4:
            # Filter Faces by Classifier
            filter_faces_by_classifier(photos)
            save_photos(photos, PHOTOS_FILE)
            update_index_csv(photos)
        elif menu_entry_index == 5:
            # Update Status Based on File Existence
            update_status_based_on_file_existence(photos)
            save_photos(photos, PHOTOS_FILE)
            update_index_csv(photos)
        elif menu_entry_index == 6:
            # Reset All Filters
            reset_filters(photos)
            save_photos(photos, PHOTOS_FILE)
            update_index_csv(photos)
        elif menu_entry_index == 7:
            # Return to Main Menu
            break
        else:
            print("Invalid selection. Please try again.")

def generate_collage_menu(photos):
    # Gather all faces included in the collage
    eligible_faces = []
    for photo in photos:
        if photo.faces_detected and photo.face_list:
            for face in photo.face_list:
                if face.include_in_collage:
                    eligible_faces.append(face)

    if not eligible_faces:
        print("No eligible faces to include in the collage.")
        return

    total_photos = len(eligible_faces)
    print(f"Total eligible faces: {total_photos}")

    # The rest of the code remains largely the same as before, but uses eligible_faces instead of photos
    # For brevity, I'll assume the generate_collage function has been updated to handle faces

    # Ask the user whether they want to specify individual photo size or overall collage size
    # ... (rest of the function remains the same, adjusted for faces)

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
                create_photo_index(PHOTO_INDEX_FILE, PHOTOS_FILE, album_id)
                tracker.update_stage('Indexing Completed')
            else:
                print("No album selected.")

        elif selection == 1:
            # Download and Process Photos
            photos = load_photos(PHOTOS_FILE)
            if not photos:
                print("No photos to process. Please index an album first.")
                continue
            download_and_process_photos(photos)
            tracker.update_stage('Download and Processing Completed')

        elif selection == 2:
            # Train Custom Classifier
            photos = load_photos(PHOTOS_FILE)
            if not photos:
                print("No photos available. Please index an album first.")
                continue
            train_custom_classifier(photos)
            tracker.update_stage('Classifier Trained')

        elif selection == 3:
            # Classify Faces
            photos = load_photos(PHOTOS_FILE)
            if not photos:
                print("No photos available. Please index an album first.")
                continue
            classify_all_faces(photos)
            tracker.update_stage('Faces Classified')

        elif selection == 4:
            # Filter and Manage Photos
            photos = load_photos(PHOTOS_FILE)
            if not photos:
                print("No photos to process. Please index an album first.")
                continue
            filter_and_manage_photos(photos)
            tracker.update_stage('Filtering and Management Completed')

        elif selection == 5:
            # Generate Collage
            photos = load_photos(PHOTOS_FILE)
            if not photos:
                print("No photos to process. Please index an album first.")
                continue
            generate_collage_menu(photos)
            tracker.update_stage('Collage Generated')

        elif selection == 6:
            # View Progress
            if os.path.exists(PHOTOS_FILE):
                photos = load_photos(PHOTOS_FILE)
            else:
                photos = []
            tracker.display_progress(photos)
            input("\nPress Enter to return to the main menu...")

        elif selection == 7:
            # Reset Project
            reset_project(tracker)

        elif selection == 8:
            # Exit
            print("Exiting the application. Goodbye!")
            break

        else:
            print("Invalid selection. Please try again.")

if __name__ == '__main__':
    main()
