import os
import logging
from simple_term_menu import TerminalMenu
import pandas as pd
from tqdm import tqdm
import sys

from utils.google_photos_api import (
    list_albums,
    get_album_photos,
    download_photo
)
from utils.face_detection import process_images_batch
from utils.collage_utils import generate_collage
from utils.progress_tracker import ProgressTracker
from config import *

# Suppress TensorFlow and Mediapipe logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs (0 = all, 1 = INFO, 2 = WARNING, 3 = ERROR)

# Suppress absl logging from TensorFlow and Mediapipe
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except ImportError:
    pass  # absl not installed

def setup_logging():
    logging.basicConfig(
        filename='app.log',
        level=logging.ERROR,  # Set to ERROR to reduce verbosity
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def create_directories():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_IMAGES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs('credentials', exist_ok=True)

def main_menu():
    options = [
        "1. List Albums",
        "2. Index Photos in Album",
        "3. Download and Process Photos",
        "4. Generate Collage",
        "5. View Progress",
        "6. Reset Project",
        "7. Exit"
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

def create_photo_index(index_file, album_id):
    """
    Create an index of all photos in the specified album.
    """
    media_items = get_album_photos(album_id)
    if not media_items:
        logging.error("No media items found in the album.")
        return

    media_data = []
    for item in media_items:
        if 'image' in item['mimeType']:
            # Flatten the dictionary for easy DataFrame creation
            media_data.append({
                'id': item.get('id'),
                'filename': item.get('filename'),
                'description': item.get('description'),
                'mimeType': item.get('mimeType'),
                'productUrl': item.get('productUrl'),
                'baseUrl': item.get('baseUrl'),
                'creationTime': item.get('mediaMetadata', {}).get('creationTime'),
                'width': item.get('mediaMetadata', {}).get('width'),
                'height': item.get('mediaMetadata', {}).get('height'),
                # ... other metadata fields
            })

    # Save index to CSV
    df = pd.DataFrame(media_data)
    df.to_csv(index_file, index=False)
    logging.info(f"Photo index saved to {index_file}")
    print(f"Indexed {len(media_data)} photos from the album.")

def download_and_process_photos(index_file, processed_images_dir, batch_size=BATCH_SIZE, tracker=None):
    """
    Download and process photos in batches.
    """
    # Load index
    df = pd.read_csv(index_file)
    total_photos = len(df)
    logging.info(f"Total photos to process: {total_photos}")

    # Calculate total batches
    total_batches = (total_photos + batch_size - 1) // batch_size
    processed_batches = tracker.processed_batches if tracker else 0

    # Update total_batches in tracker
    if tracker:
        tracker.total_batches = total_batches
        tracker.save_progress()

    batches = [df[i:i+batch_size] for i in range(0, total_photos, batch_size)]

    for batch_num, batch_df in enumerate(batches):
        if batch_num < processed_batches:
            continue  # Skip already processed batches

        logging.info(f"Processing batch {batch_num + 1}/{total_batches}")

        # Filter out already processed photos
        processed_filenames = set(os.listdir(processed_images_dir))
        processed_filenames = {os.path.splitext(f)[0] for f in processed_filenames}
        batch_df = batch_df[~batch_df['filename'].str.replace('.', '_').isin(processed_filenames)]

        if batch_df.empty:
            logging.info("All photos in this batch have already been processed.")
            continue

        images = []
        for _, row in tqdm(batch_df.iterrows(), total=batch_df.shape[0], desc='Downloading images', file=sys.stdout):
            media_item_id = row['id']
            filename = row['filename']
            temp_image_path = download_photo(media_item_id, f"temp_{filename}", processed_images_dir)
            if temp_image_path:
                images.append((temp_image_path, filename))

        if images:
            process_images_batch(images, processed_images_dir)
            for temp_image_path, _ in images:
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
        else:
            logging.info("No images to process in this batch.")
        processed_batches += 1
        if tracker:
            tracker.update_batches(processed_batches, total_batches)

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

        # Delete index file
        if os.path.exists(INDEX_FILE):
            os.remove(INDEX_FILE)

        # Delete progress file
        if os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)

        print("Project has been reset.")
        logging.info("Project has been reset.")
    else:
        print("Project reset canceled.")

def main():
    setup_logging()
    create_directories()
    tracker = ProgressTracker(PROGRESS_FILE)

    while True:
        selection = main_menu()
        if selection == 0:
            # List Albums
            albums = list_albums()
            if albums:
                print("\nAvailable Albums:")
                for i, album in enumerate(albums):
                    print(f"{i+1}. {album.get('title', 'Untitled')} (ID: {album['id']})")
                input("\nPress Enter to return to the main menu...")
            else:
                print("No albums found.")
                input("\nPress Enter to return to the main menu...")

        elif selection == 1:
            # Index Photos in Album
            selected_album = select_album()
            if selected_album:
                album_id = selected_album['id']
                create_photo_index(INDEX_FILE, album_id)
                tracker.update_stage('Indexing Completed')
            else:
                print("No album selected.")

        elif selection == 2:
            # Download and Process Photos
            download_and_process_photos(INDEX_FILE, PROCESSED_IMAGES_DIR, batch_size=BATCH_SIZE, tracker=tracker)

        elif selection == 3:
            # Generate Collage
            generate_collage(PROCESSED_IMAGES_DIR, COLLAGE_OUTPUT_PATH)
            tracker.update_stage('Collage Generated')

        elif selection == 4:
            # View Progress
            tracker.display_progress()
            input("\nPress Enter to return to the main menu...")

        elif selection == 5:
            # Reset Project
            reset_project(tracker)

        elif selection == 6:
            # Exit
            print("Exiting the application. Goodbye!")
            break

        else:
            print("Invalid selection. Please try again.")

if __name__ == '__main__':
    main()
