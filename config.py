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
DEFAULT_DPI = 300            # Default DPI for collage generation

# Paths
DATA_DIR = os.path.join('data')
PHOTO_INDEX_FILE = os.path.join(DATA_DIR, 'photo_index.csv')
FACES_INDEX_FILE = os.path.join(DATA_DIR, 'faces_index.csv')
PHOTOS_FILE = os.path.join(DATA_DIR, 'photos.json')
PROGRESS_FILE = os.path.join(DATA_DIR, 'progress.log')
PROCESSED_IMAGES_DIR = os.path.join(DATA_DIR, 'processed_images')
ORIGINAL_IMAGES_DIR = os.path.join(DATA_DIR, 'original')
EXCLUDED_IMAGES_DIR = os.path.join(DATA_DIR, 'excluded_images')
OUTPUT_DIR = os.path.join('output')
LOG_DIR = os.path.join(DATA_DIR, 'logs')  # Directory for logs

MODEL_DIR = os.path.join(DATA_DIR, 'model')
POSITIVE_FACES_DIR = os.path.join(MODEL_DIR, 'positive')
NEGATIVE_FACES_DIR = os.path.join(MODEL_DIR, 'negative')
CLASSIFIER_MODEL_PATH = os.path.join(MODEL_DIR, 'classifier.pkl')

# Processing settings
DELETE_ORIGINAL_AFTER_PROCESSING = True  # Set to False to keep original images after processing

# Logging settings
LOG_LEVEL = 'ERROR'          # Set to 'ERROR' for minimal logs or 'DEBUG' for detailed logs
