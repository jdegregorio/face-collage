import os
import pickle
import logging
import requests
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from contextlib import contextmanager
import sys
from config import CLIENT_SECRETS_FILE, SCOPES

@contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

def authenticate_google_photos():
    """
    Authenticate the user with Google Photos and return the service object.
    """
    creds = None
    token_pickle = os.path.join('credentials', 'token.pickle')

    if os.path.exists(token_pickle):
        with open(token_pickle, 'rb') as token:
            creds = pickle.load(token)
    else:
        with suppress_stderr():
            flow = InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRETS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_pickle, 'wb') as token:
            pickle.dump(creds, token)

    with suppress_stderr():
        service = build('photoslibrary', 'v1', credentials=creds, static_discovery=False)
    return service

def list_albums():
    """
    List all albums in the user's Google Photos library.
    """
    service = authenticate_google_photos()
    albums = []
    next_page_token = None

    while True:
        try:
            with suppress_stderr():
                results = service.albums().list(
                    pageSize=50,
                    pageToken=next_page_token
                ).execute()
            items = results.get('albums', [])
            albums.extend(items)
            next_page_token = results.get('nextPageToken')
            if not next_page_token:
                break
        except Exception as e:
            logging.error(f"Error listing albums: {e}")
            break

    return albums

def get_album_photos(album_id):
    """
    Retrieve all photo media items from a specific album.
    """
    service = authenticate_google_photos()
    media_items = []
    next_page_token = None

    while True:
        try:
            body = {
                "albumId": album_id,
                "pageSize": 100,
                "pageToken": next_page_token
            }
            with suppress_stderr():
                results = service.mediaItems().search(body=body).execute()
            items = results.get('mediaItems', [])
            media_items.extend(items)
            next_page_token = results.get('nextPageToken')
            if not next_page_token:
                break
        except Exception as e:
            logging.error(f"Error retrieving media items: {e}")
            break

    return media_items

def download_photo(media_item_id, filename, download_dir):
    """
    Download a specific photo given its media item ID.
    """
    service = authenticate_google_photos()
    try:
        with suppress_stderr():
            media_item = service.mediaItems().get(mediaItemId=media_item_id).execute()
        base_url = media_item['baseUrl']
        mime_type = media_item['mimeType']

        if 'image' in mime_type:
            download_url = base_url + '=d'  # Download original quality
            response = requests.get(download_url)
            if response.status_code == 200:
                file_path = os.path.join(download_dir, filename)
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                logging.info(f"Downloaded {filename}")
                return file_path
            else:
                logging.warning(f"Failed to download {filename}: HTTP {response.status_code}")
                return None
        else:
            logging.warning(f"Media item {media_item_id} is not an image.")
            return None
    except Exception as e:
        logging.error(f"Error downloading photo {media_item_id}: {e}")
        return None
