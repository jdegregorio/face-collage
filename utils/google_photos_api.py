import os
import pickle
import requests
import logging
import pandas as pd
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from tqdm import tqdm
from config import CLIENT_SECRETS_FILE, SCOPES, BATCH_SIZE
from utils.face_detection import process_images_batch

def authenticate_google_photos():
    creds = None
    token_pickle = os.path.join('credentials', 'token.pickle')

    if os.path.exists(token_pickle):
        with open(token_pickle, 'rb') as token:
            creds = pickle.load(token)
    else:
        flow = InstalledAppFlow.from_client_secrets_file(
            CLIENT_SECRETS_FILE, SCOPES)
        creds = flow.run_local_server(port=0)
        with open(token_pickle, 'wb') as token:
            pickle.dump(creds, token)
    return creds

def create_photo_index(index_file, album_id=None):
    creds = authenticate_google_photos()
    service = build('photoslibrary', 'v1', credentials=creds, static_discovery=False)

    logging.info("Creating photo index from Google Photos album...")
    media_items = []
    next_page_token = None

    while True:
        try:
            response = service.mediaItems().list(
                pageSize=100,
                pageToken=next_page_token
            ).execute()
            items = response.get('mediaItems', [])
            if not items:
                break
            for item in items:
                if 'image' in item['mimeType']:
                    media_items.append({
                        'id': item['id'],
                        'filename': item['filename'],
                        'baseUrl': item['baseUrl'],
                        'mimeType': item['mimeType']
                    })
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
        except Exception as e:
            logging.error(f"Error retrieving media items: {e}")
            break

    # Save index to CSV
    df = pd.DataFrame(media_items)
    df.to_csv(index_file, index=False)
    logging.info(f"Photo index saved to {index_file}")

def download_and_process_batch(index_file, processed_images_dir, batch_size=BATCH_SIZE, tracker=None):
    creds = authenticate_google_photos()
    service = build('photoslibrary', 'v1', credentials=creds, static_discovery=False)

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
            base_url = row['baseUrl']
            download_url = base_url + '=d'  # Download original quality

            try:
                response = requests.get(download_url)
                if response.status_code == 200:
                    # Save the image temporarily
                    temp_image_path = os.path.join(processed_images_dir, f"temp_{filename}")
                    with open(temp_image_path, 'wb') as f:
                        f.write(response.content)
                    images.append((temp_image_path, filename))
                else:
                    logging.warning(f"Failed to download {filename}: HTTP {response.status_code}")
            except Exception as e:
                logging.error(f"Error downloading {filename}: {e}")

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
