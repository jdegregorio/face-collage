import os

# Google Photos API settings
CLIENT_SECRETS_FILE = os.path.join('credentials', 'client_secrets.json')
SCOPES = ['https://www.googleapis.com/auth/photoslibrary.readonly']
NGROK_URL = ' https://88a6-73-157-10-120.ngrok-free.app'  # Replace with your actual ngrok URL

# Image processing settings
MIN_FACE_SIZE = 100          # Minimum face size in pixels to consider
IMAGE_SIZE = 256             # Size to which each face image will be resized (pixels)
MAX_YAW = 45.0               # Maximum yaw angle to include (degrees)
MAX_PITCH = 45.0             # Maximum pitch angle to include (degrees)

# Collage settings
COLLAGE_WIDTH = 6000         # Width of the final collage image in pixels
COLLAGE_HEIGHT = 9000        # Height of the final collage image in pixels

# Paths
DATA_DIR = os.path.join('data')
INDEX_FILE = os.path.join(DATA_DIR, 'index.csv')
PROGRESS_FILE = os.path.join(DATA_DIR, 'progress.log')
PROCESSED_IMAGES_DIR = os.path.join(DATA_DIR, 'processed_images')
OUTPUT_DIR = os.path.join('output')
COLLAGE_OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'final_collage.jpg')

# Batch processing settings
BATCH_SIZE = 30              # Number of photos to process in each batch

# Logging settings
LOG_LEVEL = 'INFO'           # Can be set to 'DEBUG' for more detailed logs
