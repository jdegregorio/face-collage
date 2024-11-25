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
        "3. Perform Head Pose and Facial Features Estimation",
        "4. Train Custom Classifier",
        "5. Classify Faces",
        "6. Filter and Manage Photos",
        "7. Generate Collage",
        "8. View Progress",
        "9. Reset Project",
        "10. Exit"
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

def perform_head_pose_and_facial_features_estimation(photos):
    """
    Perform head pose estimation and extract facial features on processed faces.
    """
    total_faces = sum(len(photo.face_list) for photo in photos if photo.face_list)
    logging.info(f"Total faces to process: {total_faces}")

    faces_to_process = []
    for photo in photos:
        if photo.face_list:
            for face in photo.face_list:
                if face.head_pose_estimation_status != 'success':
                    faces_to_process.append(face)

    if not faces_to_process:
        print("All faces have head pose estimation completed.")
        return

    from utils.head_pose_and_facial_features import estimate_head_pose_and_facial_features

    for face in tqdm(faces_to_process, desc="Estimating head poses and extracting facial features", unit="face"):
        try:
            result = estimate_head_pose_and_facial_features(face.image_path)
            if result:
                yaw, pitch, roll, facial_features = result
                face.yaw = yaw
                face.pitch = pitch
                face.roll = roll
                face.head_pose_estimation_status = 'success'
                face.head_pose_estimation_error = ''
                # Update facial features
                face.left_eye_openness = facial_features['left_eye_openness']
                face.right_eye_openness = facial_features['right_eye_openness']
                face.avg_eye_openness = facial_features['avg_eye_openness']
                face.mouth_openness = facial_features['mouth_openness']
                face.facial_features_status = 'success'
            else:
                face.head_pose_estimation_status = 'failed'
                face.head_pose_estimation_error = 'Head pose estimation failed'
                face.facial_features_status = 'failed'
                logging.warning(f"Failed to estimate head pose for face {face.id}")
            # Save progress after each face
            save_photos(photos, PHOTOS_FILE)
            update_index_csv(photos)

        except Exception as e:
            logging.error(f"Unexpected error processing face {face.id}: {e}")
            face.head_pose_estimation_status = 'failed'
            face.head_pose_estimation_error = 'Unexpected error'
            face.facial_features_status = 'failed'
            # Save progress and continue to next face
            save_photos(photos, PHOTOS_FILE)
            update_index_csv(photos)
            continue  # Skip to next face

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
            "1. Exclude Faces with Failed Processing",
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
            # Exclude Faces with Failed Processing
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
    eligible_faces = []
    for photo in photos:
        if photo.face_list:
            for face in photo.face_list:
                if face.include_in_collage:
                    # Assign photo timestamp to face if not already assigned
                    if not face.timestamp:
                        face.timestamp = photo.timestamp
                    eligible_faces.append(face)

    if not eligible_faces:
        print("No eligible faces to include in the collage.")
        return

    total_faces = len(eligible_faces)
    print(f"Total eligible faces: {total_faces}")

    # Ask the user whether they want to specify individual face image size or overall collage size
    options = [
        "1. Specify individual face image size",
        "2. Specify overall collage size",
        "3. Cancel and return to main menu"
    ]
    terminal_menu = TerminalMenu(options, title="Collage Configuration")
    menu_entry_index = terminal_menu.show()

    if menu_entry_index == 2 or menu_entry_index is None:
        return

    dpi_default = DEFAULT_DPI
    print(f"Default DPI is {dpi_default}.")
    dpi_input = input(f"Enter desired DPI (dots per inch) [default: {dpi_default}]: ")
    dpi = int(dpi_input) if dpi_input.strip() else dpi_default

    if menu_entry_index == 0:
        # User wants to specify individual face image size
        # Present standard sizes or allow custom
        standard_face_sizes = ["1x1", "2x2", "3x3", "4x4", "5x5"]
        options_face_size = standard_face_sizes + ["Custom size"]
        terminal_menu = TerminalMenu(options_face_size, title="Select Individual Face Image Size (in inches)")
        size_index = terminal_menu.show()

        if size_index == len(options_face_size) - 1 or size_index is None:
            # Custom size
            width_input = input("Enter custom face image width (in inches): ")
            height_input = input("Enter custom face image height (in inches): ")
            try:
                face_width_in = float(width_input)
                face_height_in = float(height_input)
            except ValueError:
                print("Invalid input. Please enter numeric values.")
                return
        else:
            size_str = standard_face_sizes[size_index]
            face_width_in, face_height_in = map(float, size_str.split('x'))

        # Now, suggest standard collage sizes that fit perfectly with the face image size
        standard_print_sizes = [
            "4x6", "5x7", "8x10", "11x14", "12x18", "16x20", "20x30", "4x4", "8x8", "12x12", "12x36"
        ]
        possible_collage_options = []
        for size_str in standard_print_sizes:
            collage_width_in, collage_height_in = map(float, size_str.split('x'))
            for orientation in ['portrait', 'landscape']:
                cw_in, ch_in = collage_width_in, collage_height_in  # Copy original values
                if orientation == 'portrait' and cw_in > ch_in:
                    cw_in, ch_in = ch_in, cw_in
                elif orientation == 'landscape' and cw_in < ch_in:
                    cw_in, ch_in = ch_in, cw_in
                # Check if the face image size fits perfectly into the collage size
                cols = cw_in / face_width_in
                rows = ch_in / face_height_in
                if cols.is_integer() and rows.is_integer():
                    cols = int(cols)
                    rows = int(rows)
                    total_cells = rows * cols
                    if total_cells <= total_faces:
                        possible_collage_options.append((size_str, orientation, rows, cols, cw_in, ch_in))
        if not possible_collage_options:
            print("No standard collage sizes fit your face image size and number of faces.")
            # Optionally, allow user to enter custom collage size
            custom_size_input = input("Do you want to enter a custom collage size? (yes/no): ")
            if custom_size_input.lower() == 'yes':
                width_input = input("Enter custom collage width (in inches): ")
                height_input = input("Enter custom collage height (in inches): ")
                try:
                    collage_width_in = float(width_input)
                    collage_height_in = float(height_input)
                except ValueError:
                    print("Invalid input. Please enter numeric values.")
                    return
                # Check if the face image size fits perfectly into the collage size
                cols = collage_width_in / face_width_in
                rows = collage_height_in / face_height_in
                if cols.is_integer() and rows.is_integer():
                    cols = int(cols)
                    rows = int(rows)
                    total_cells = rows * cols
                    if total_cells <= total_faces:
                        # Proceed with this configuration
                        orientation = 'custom'
                    else:
                        print("Not enough faces to fill this collage size.")
                        return
                else:
                    print("Face image size does not fit perfectly into the collage size.")
                    return
            else:
                return
        else:
            print("\nPossible collage sizes based on your face image size:")
            options_collage = []
            for i, (size_str, orientation, rows, cols, cw_in, ch_in) in enumerate(possible_collage_options):
                options_collage.append(f"{i+1}. {size_str} ({orientation}), {rows} rows x {cols} columns")
            terminal_menu = TerminalMenu(options_collage, title="Select a Collage Size Option")
            selection_index = terminal_menu.show()
            if selection_index is None:
                return
            size_str, orientation, rows, cols, collage_width_in, collage_height_in = possible_collage_options[selection_index]

        # Now, proceed with the selected collage configuration

    elif menu_entry_index == 1:
        # User wants to specify overall collage size
        # Present standard print sizes or allow custom
        standard_print_sizes = [
            "4x6", "5x7", "8x10", "11x14", "12x18", "16x20", "20x30", "4x4", "8x8", "12x12", "12x36"
        ]
        options_print_size = standard_print_sizes + ["Custom size"]
        terminal_menu = TerminalMenu(options_print_size, title="Select Overall Collage Size (in inches)")
        size_index = terminal_menu.show()

        if size_index == len(options_print_size) - 1 or size_index is None:
            # Custom size
            width_input = input("Enter custom collage width (in inches): ")
            height_input = input("Enter custom collage height (in inches): ")
            try:
                collage_width_in = float(width_input)
                collage_height_in = float(height_input)
            except ValueError:
                print("Invalid input. Please enter numeric values.")
                return
        else:
            size_str = options_print_size[size_index]
            collage_width_in, collage_height_in = map(float, size_str.split('x'))

        # Ask for orientation if width != height
        if collage_width_in != collage_height_in:
            options_orientation = ["1. Portrait", "2. Landscape"]
            terminal_menu = TerminalMenu(options_orientation, title="Select Collage Orientation")
            orientation_index = terminal_menu.show()

            if orientation_index == 0:
                # Portrait
                if collage_width_in > collage_height_in:
                    collage_width_in, collage_height_in = collage_height_in, collage_width_in
            elif orientation_index == 1:
                # Landscape
                if collage_width_in < collage_height_in:
                    collage_width_in, collage_height_in = collage_height_in, collage_width_in

        # Now, we need to calculate possible face image sizes and grid dimensions
        # Ask user to input desired number of rows or columns, or individual face image size
        options_grid = [
            "1. Specify individual face image size",
            "2. Specify number of rows",
            "3. Specify number of columns",
            "4. Cancel and return to main menu"
        ]
        terminal_menu = TerminalMenu(options_grid, title="Specify Grid Configuration")
        grid_config_index = terminal_menu.show()

        if grid_config_index == 3 or grid_config_index is None:
            return

        if grid_config_index == 0:
            # Specify individual face image size
            width_input = input("Enter individual face image width (in inches): ")
            height_input = input("Enter individual face image height (in inches): ")
            try:
                face_width_in = float(width_input)
                face_height_in = float(height_input)
            except ValueError:
                print("Invalid input. Please enter numeric values.")
                return
            cols = collage_width_in / face_width_in
            rows = collage_height_in / face_height_in
            if not cols.is_integer() or not rows.is_integer():
                print("Face image size does not fit perfectly into the collage size.")
                return
            cols = int(cols)
            rows = int(rows)
            total_cells = rows * cols
            if total_cells > total_faces:
                print(f"Not enough faces to fill the grid ({total_cells} cells needed, but only {total_faces} faces available).")
                return
        elif grid_config_index == 1:
            # Specify number of rows
            rows_input = input("Enter number of rows: ")
            try:
                rows = int(rows_input)
            except ValueError:
                print("Invalid input. Please enter an integer.")
                return
            face_height_in = collage_height_in / rows
            face_width_in = face_height_in  # Assuming square images
            cols = collage_width_in / face_width_in
            if not cols.is_integer():
                print("The number of columns is not an integer with the given number of rows and collage dimensions.")
                return
            cols = int(cols)
            total_cells = rows * cols
            if total_cells > total_faces:
                print(f"Not enough faces to fill the grid ({total_cells} cells needed, but only {total_faces} faces available).")
                return
        elif grid_config_index == 2:
            # Specify number of columns
            cols_input = input("Enter number of columns: ")
            try:
                cols = int(cols_input)
            except ValueError:
                print("Invalid input. Please enter an integer.")
                return
            face_width_in = collage_width_in / cols
            face_height_in = face_width_in  # Assuming square images
            rows = collage_height_in / face_height_in
            if not rows.is_integer():
                print("The number of rows is not an integer with the given number of columns and collage dimensions.")
                return
            rows = int(rows)
            total_cells = rows * cols
            if total_cells > total_faces:
                print(f"Not enough faces to fill the grid ({total_cells} cells needed, but only {total_faces} faces available).")
                return

    else:
        return

    # Calculate total number of faces needed
    total_faces_needed = rows * cols

    if total_faces_needed > total_faces:
        print(f"\nYou need {total_faces_needed} faces to fill the grid, but only have {total_faces} eligible faces.")
        print("Please adjust your settings or include more faces.")
        return
    elif total_faces_needed < total_faces:
        print(f"\nThe grid will use {total_faces_needed} faces, but you have {total_faces} eligible faces.")
        print("Excess faces will be sampled to fit the grid.")
        # Implement sampling strategy to select faces
        selected_faces = sample_photos_temporally(eligible_faces, total_faces_needed)
    else:
        selected_faces = eligible_faces

    # Ask for wrap-around method
    options_wrap = [
        "1. Left to right, then down (like reading a book)",
        "2. Left to right, then down, alternating directions (snaking)",
        "3. Top to bottom, then right",
        "4. Top to bottom, then right, alternating directions (snaking)",
        "5. Cancel and return to main menu"
    ]
    terminal_menu = TerminalMenu(options_wrap, title="Select Wrap-around Method")
    wrap_method_index = terminal_menu.show()

    if wrap_method_index == 4 or wrap_method_index is None:
        return

    wrap_methods = {
        0: 'lr_down',
        1: 'lr_down_snake',
        2: 'tb_right',
        3: 'tb_right_snake'
    }

    wrap_method = wrap_methods.get(wrap_method_index)

    # Now, calculate the pixel dimensions
    collage_width_px = int(collage_width_in * dpi)
    collage_height_px = int(collage_height_in * dpi)
    face_width_px = int(face_width_in * dpi)
    face_height_px = int(face_height_in * dpi)

    # Ask user for output filename
    output_filename = input("Enter a name for the output collage file (without extension): ").strip()
    if not output_filename:
        print("Filename cannot be empty.")
        return
    output_path = os.path.join(OUTPUT_DIR, f"{output_filename}.jpg")

    # Prepare options to pass to generate_collage
    collage_options = {
        'collage_width_px': collage_width_px,
        'collage_height_px': collage_height_px,
        'photo_width_px': face_width_px,
        'photo_height_px': face_height_px,
        'rows': rows,
        'cols': cols,
        'wrap_method': wrap_method
    }

    # Call generate_collage
    generate_collage(selected_faces, output_path, collage_options)
    print(f"\nCollage generated and saved to {output_path}")

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
            # Perform Head Pose and Facial Features Estimation
            photos = load_photos(PHOTOS_FILE)
            if not photos:
                print("No photos to process. Please index an album first.")
                continue
            perform_head_pose_and_facial_features_estimation(photos)
            tracker.update_stage('Head Pose and Facial Features Estimation Completed')

        elif selection == 3:
            # Train Custom Classifier
            photos = load_photos(PHOTOS_FILE)
            if not photos:
                print("No photos available. Please index an album first.")
                continue
            train_custom_classifier(photos)
            tracker.update_stage('Classifier Trained')

        elif selection == 4:
            # Classify Faces
            photos = load_photos(PHOTOS_FILE)
            if not photos:
                print("No photos available. Please index an album first.")
                continue
            classify_all_faces(photos)
            tracker.update_stage('Faces Classified')

        elif selection == 5:
            # Filter and Manage Photos
            photos = load_photos(PHOTOS_FILE)
            if not photos:
                print("No photos to process. Please index an album first.")
                continue
            filter_and_manage_photos(photos)
            tracker.update_stage('Filtering and Management Completed')

        elif selection == 6:
            # Generate Collage
            photos = load_photos(PHOTOS_FILE)
            if not photos:
                print("No photos to process. Please index an album first.")
                continue
            generate_collage_menu(photos)
            tracker.update_stage('Collage Generated')

        elif selection == 7:
            # View Progress
            if os.path.exists(PHOTOS_FILE):
                photos = load_photos(PHOTOS_FILE)
            else:
                photos = []
            tracker.display_progress(photos)
            input("\nPress Enter to return to the main menu...")

        elif selection == 8:
            # Reset Project
            reset_project(tracker)

        elif selection == 9:
            # Exit
            print("Exiting the application. Goodbye!")
            break

        else:
            print("Invalid selection. Please try again.")

if __name__ == '__main__':
    main()
