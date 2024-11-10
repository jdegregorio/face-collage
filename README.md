
# **Super Collage**

This project is designed to create a unique, large-scale photo collage from images in a Google Photos album. The collage arranges images based on the angle of the subject's face and the date of capture, creating a visual timeline of facial orientations over time. The project is particularly suited for creating collages of individuals, capturing the subtle changes in expression and pose over a period.

## **Table of Contents**
- [**Super Collage**](#super-collage)
  - [**Table of Contents**](#table-of-contents)
  - [**Project Overview**](#project-overview)
  - [**Features**](#features)
  - [**Project Directory Structure**](#project-directory-structure)
  - [**Prerequisites**](#prerequisites)
  - [**Installation**](#installation)
  - [**Usage**](#usage)
    - [**Running the Application**](#running-the-application)
    - [**Workflow Steps**](#workflow-steps)
    - [**Resuming a Project in Progress**](#resuming-a-project-in-progress)
  - [**Configuration**](#configuration)
  - [**Troubleshooting**](#troubleshooting)
  - [**License**](#license)

---

## **Project Overview**

The Super Collage project automates the process of downloading, processing, and organizing photos from a Google Photos album to create a customized photo collage. The collage is based on specific parameters such as face orientation and the timestamp of each image, resulting in a visually organized timeline.

This project provides a user-friendly command-line interface that guides you through each step, allowing you to pause and resume the process at any point.

---

## **Features**

- **Automated Photo Indexing**: Retrieve and index photo metadata from a Google Photos album.
- **Photo Downloading, Cropping, and Resizing**: Download, crop, and resize images for face-centered compositions.
- **Face Detection and Pose Estimation**: Detect faces in photos and estimate head pose angles for collage organization.
- **Collage Generation**: Create a high-resolution collage arranged by head pose and time.
- **Progress Tracking**: Track and resume project stages with a built-in progress tracker.
- **User-Friendly Interface**: A simple command-line menu to navigate through the project steps.

---

## **Project Directory Structure**

Here’s an overview of the project directory structure:

```
super_collage_project/
├── main.py                      # Main script to run the application
├── config.py                    # Configuration settings
├── requirements.txt             # Dependencies
├── README.md                    # Project documentation
├── data/
│   ├── photos.json              # Serialized metadata and processing state for each photo
│   ├── index.csv                # CSV report of the latest photo statuses
│   ├── original/                # Original downloaded images
│   ├── processed_images/        # Cropped and resized face images
│   └── progress.log             # Log file to track project progress
├── output/
│   └── final_collage.jpg        # The resulting collage image
└── utils/                       # Helper scripts
    ├── google_photos_api.py     # Google Photos API interface
    ├── face_detection.py        # Functions for face detection and cropping
    ├── head_pose_estimation.py  # Functions for head pose estimation
    ├── collage_utils.py         # Collage creation functions
    ├── photo.py                 # Custom class for photo metadata and processing state
    └── progress_tracker.py      # Progress tracking utility
```

---

## **Prerequisites**

- **Python** 3.7 or higher.
- A **Google API** project with OAuth 2.0 credentials (`client_secrets.json` file) for accessing Google Photos.
- OpenCV, MediaPipe, and other required libraries as listed in `requirements.txt`.

---

## **Installation**

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/super_collage_project.git
   ```

2. **Navigate to the project directory**:

   ```bash
   cd super_collage_project
   ```

3. **Install the required packages**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Google API credentials**:

   - Go to [Google Cloud Console](https://console.cloud.google.com/).
   - Create a project and enable the Google Photos API.
   - Create OAuth 2.0 credentials and download the `client_secrets.json` file.
   - Place the `client_secrets.json` file in the `credentials` directory.

---

## **Usage**

### **Running the Application**

Run the main script to launch the application’s command-line interface:

```bash
python main.py
```

### **Workflow Steps**

Once the application is launched, you’ll see a menu-driven interface that guides you through each stage of the project. The steps are as follows:

1. **Index Photos in Album**:

   - Select this option to index all photos in the Google Photos album.
   - This step retrieves metadata (photo ID, filename, base URL, etc.) and saves it in `data/photos.json`.
   - A CSV report of the current statuses will also be created in `data/index.csv`.

2. **Download, Crop & Resize Photos**:

   - Downloads photos one at a time, detects faces, centers and resizes each image.
   - Processed photos are saved in `data/processed_images/`.
   - Progress is saved in `data/photos.json`, allowing you to resume if the operation is interrupted.

3. **Perform Head Pose Estimation**:

   - After downloading, cropping, and resizing, select this option to analyze head pose.
   - The system will estimate yaw, pitch, and roll angles for each face image and save this data in `data/photos.json`.

4. **Generate Collage**:

   - After head pose estimation, this option arranges the images into a collage.
   - Images are placed on a grid based on head orientation (x-axis) and time (y-axis).
   - The collage is saved to `output/final_collage.jpg`.

5. **View Progress**:

   - Displays the current progress and stage of the project.

6. **Reset Project**:

   - Resets the project, deleting all progress and processed data.

7. **Exit**:

   - Exits the application.

### **Resuming a Project in Progress**

The application is designed to support resumability. Progress is tracked in `data/photos.json`, allowing you to start, stop, and resume the project as needed. Here’s how to resume from any point:

- **If Indexing is Completed**:
  - Skip indexing and proceed to the "Download, Crop & Resize Photos" step.

- **If Download, Crop & Resize was Interrupted**:
  - Select "Download, Crop & Resize Photos" to continue from where it left off.
  - The tracker will resume based on the data saved in `photos.json`.

- **If Head Pose Estimation was Interrupted**:
  - Select "Perform Head Pose Estimation" to continue from the last completed image.

- **If Ready to Generate the Collage**:
  - Ensure all photos have completed head pose estimation, then select "Generate Collage".

---

## **Configuration**

All key parameters and settings are stored in `config.py`. You can adjust the following:

- **Delete Original Images**: Set `DELETE_ORIGINAL_AFTER_PROCESSING = True` to delete the raw photos after processing.
- **Image Size**: Controls the size to which each face image will be resized (default: 256x256).
- **Collage Dimensions**: Width and height of the final collage image.
- **Logging Level**: Set to `DEBUG` for detailed logs, or `INFO` for standard logs.

---

## **Troubleshooting**

- **Google Photos Access Issues**:
  - Ensure the `client_secrets.json` file is correctly placed in the `credentials` folder.
  - Verify that the Google Photos API is enabled in your Google Cloud project.

- **Progress Log Not Updating**:
  - Make sure the application has write access to the `data/` directory.
  - Check for any error messages in `progress.log`.

- **Insufficient Disk Space**:
  - To manage storage, set `DELETE_ORIGINAL_AFTER_PROCESSING = True` to remove original photos after processing.

---

## **License**

This project is licensed under the GNU General Public License v3.0.

---

This README should help you get started with the project and understand the steps involved. Let me know if you need any further customization!