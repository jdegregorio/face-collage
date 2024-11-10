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

The Image Collage Project automates the process of downloading, processing, and organizing photos from a Google Photos album to create a customized photo collage. The collage is based on specific parameters such as face orientation and the timestamp of each image, resulting in a visually organized timeline. 

This project provides a user-friendly command-line interface that guides you through each step, allowing you to pause and resume the process at any point.

---

## **Features**

- **Automated Photo Indexing**: Retrieve and index photo metadata from a Google Photos album.
- **Batch Photo Processing**: Download, crop, resize, and analyze images in manageable batches to optimize memory and storage.
- **Face Detection and Pose Estimation**: Detect faces in photos and estimate head pose angles for collage organization.
- **Collage Generation**: Create a high-resolution collage arranged by head pose and time.
- **Progress Tracking**: Track and resume project stages with a built-in progress tracker.
- **User-Friendly Interface**: A simple command-line menu to navigate through the project steps.

---

## **Project Directory Structure**

Here’s an overview of the project directory structure:

```
image_collage_project/
├── main.py
├── config.py
├── requirements.txt
├── README.md
├── utils/
│   ├── __init__.py
│   ├── google_photos_api.py
│   ├── face_detection.py
│   ├── head_pose_estimation.py
│   ├── image_utils.py
│   ├── collage_utils.py
│   ├── progress_tracker.py
├── data/
│   ├── index.csv              # Index of all photos in the album
│   ├── progress.log           # Log file to track progress
│   ├── processed_images/      # Cropped and aligned face images
├── output/
│   └── final_collage.jpg      # The resulting collage image
└── credentials/
    └── client_secrets.json    # Google API client secrets
```

---

## **Prerequisites**

- Python 3.7 or higher.
- A Google API project with OAuth 2.0 credentials (client_secrets.json file) for accessing Google Photos.

---

## **Installation**

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/image_collage_project.git
   ```

2. **Navigate to the project directory**:

   ```bash
   cd image_collage_project
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

1. **Index Photos**:

   - Select this option to index all photos in the Google Photos album.
   - This step retrieves metadata (photo ID, filename, base URL) and saves it in `data/index.csv`.
   - If you’ve previously indexed photos, you can skip this step.

2. **Process Photos in Batches**:

   - Downloads and processes photos in manageable batches (default batch size is 30).
   - Each photo is downloaded, cropped around the face, resized, and analyzed for face orientation.
   - If the process is interrupted, you can resume by selecting this option again.
   - Processed photos are saved in `data/processed_images/`, and progress is tracked in `data/progress.log`.

3. **Generate Collage**:

   - After processing the photos, use this option to create the final collage.
   - The collage image is generated based on face orientation (x-axis) and time (y-axis) and saved to `output/final_collage.jpg`.

4. **View Progress**:

   - Displays the current progress and the stage of the project.
   - This option is useful if you need to check how much work has been completed or what remains.

5. **Exit**:

   - Exits the application.

### **Resuming a Project in Progress**

The application is designed to support resumability. Progress is tracked in `data/progress.log`, allowing you to start, stop, and resume the project as needed. Here’s how to resume from any point:

- **If Indexing is Completed**:
  - Skip indexing and proceed to batch processing.

- **If Batch Processing was Interrupted**:
  - Select “Process Photos in Batches” to continue from the last processed batch.
  - The tracker will pick up where it left off.

- **If Ready to Generate the Collage**:
  - Ensure batch processing is complete, then select “Generate Collage”.

---

## **Configuration**

All key parameters and settings are stored in `config.py`. You can adjust the following:

- **Batch Size**: The number of photos to download and process at a time.
- **Image Dimensions**: Size to which each face image will be resized (default: 256x256).
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
  - To manage storage, the program deletes original high-resolution photos after processing.
  - Adjust the `BATCH_SIZE` in `config.py` if memory or storage constraints arise.

---

## **License**

This project is licensed under the GNU 3.0 License.
