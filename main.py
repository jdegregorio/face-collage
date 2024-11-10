# main.py

import os
import logging
from simple_term_menu import TerminalMenu
import pandas as pd

from utils.google_photos_api import (
    list_albums,
    get_album_photos,
    download_photo
)
from utils.face_detection import process_images_batch
from utils.collage_utils import generate_collage
from utils.progress_tracker import ProgressTracker
from config import *

def setup_logging():
    logging.basicConfig(
        filename=PROGRESS_FILE,
        level=logging.DEBUG,
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
        "6. Exit"
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
                'photo_cameraMake': item.get('mediaMetadata', {}).get('photo', {}).get('cameraMake'),
                'photo_cameraModel': item.get('mediaMetadata', {}).get('photo', {}).get('cameraModel'),
                'photo_focalLength': item.get('mediaMetadata', {}).get('photo', {}).get('focalLength'),
                'photo_apertureFNumber': item.get('mediaMetadata', {}).get('photo', {}).get('apertureFNumber'),
                'photo_isoEquivalent': item.get('mediaMetadata', {}).get('photo', {}).get('isoEquivalent'),
                'video_cameraMake': item.get('mediaMetadata', {}).get('video', {}).get('cameraMake'),
                'video_cameraModel': item.get('mediaMetadata', {}).get('video', {}).get('cameraModel'),
                'video_fps': item.get('mediaMetadata', {}).get('video', {}).get('fps'),
                'video_status': item.get('mediaMetadata', {}).get('video', {}).get('status')
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
        for _, row in batch_df.iterrows():
            media_item_id = row['id']
            filename = row['filename']
            temp_image_path = download_photo(media_item_id, f"temp_{filename}", processed_images_dir)
            if temp_image_path:
                images.append((temp_image_path, filename))

        # Process images
        if images:
            process_images_batch(images, processed_images_dir)
            # Remove temporary images
            for temp_image_path, _ in images:
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
        else:
            logging.info("No images to process in this batch.")

        processed_batches += 1
        if tracker:
            tracker.update_batches(processed_batches, total_batches)

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
            # Exit
            print("Exiting the application. Goodbye!")
            break

        else:
            print("Invalid selection. Please try again.")

if __name__ == '__main__':
    main()
