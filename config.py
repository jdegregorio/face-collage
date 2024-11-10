import os

# Google Photos API settings
CLIENT_SECRETS_FILE = os.path.join('credentials', 'client_secrets.json')
SCOPES = ['https://www.googleapis.com/auth/photoslibrary.readonly']

# Image processing settings
MIN_FACE_SIZE = 100          # Minimum face size in pixels to consider
IMAGE_SIZE = 256             # Size to which each face image will be resized (pixels)
MAX_YAW = 45.0               # Maximum yaw angle to include (degrees)
MAX_PITCH = 45.0             # Maximum pitch angle to include (degrees)

# Bounding box expansion settings
INITIAL_BBOX_EXPANSION = 2   # Percentage to expand the initial bounding box (e.g., 0.2 for 20%)

# Collage settings
COLLAGE_WIDTH = 6000         # Width of the final collage image in pixels
COLLAGE_HEIGHT = 9000        # Height of the final collage image in pixels

# Paths
DATA_DIR = os.path.join('data')
INDEX_FILE = os.path.join(DATA_DIR, 'index.csv')
PHOTOS_FILE = os.path.join(DATA_DIR, 'photos.json')
PROGRESS_FILE = os.path.join(DATA_DIR, 'progress.log')
PROCESSED_IMAGES_DIR = os.path.join(DATA_DIR, 'processed_images')
ORIGINAL_IMAGES_DIR = os.path.join(DATA_DIR, 'original')
OUTPUT_DIR = os.path.join('output')
COLLAGE_OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'final_collage.jpg')

# Processing settings
DELETE_ORIGINAL_AFTER_PROCESSING = True  # Set to False to keep original images after processing

# Logging settings
LOG_LEVEL = 'ERROR'          # Set to 'ERROR' to reduce verbosity
