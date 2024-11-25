import os
import numpy as np
import joblib
from tqdm import tqdm
import logging
from sklearn.svm import SVC
from utils.face import Face
from config import CLASSIFIER_MODEL_PATH
from utils.head_pose_and_facial_features import extract_face_embedding

def train_classifier(positive_dir, negative_dir, classifier_path):
    """
    Train a classifier using the positive and negative face images.
    """
    # Collect training data
    X = []
    y = []

    # Positive samples
    for filename in os.listdir(positive_dir):
        filepath = os.path.join(positive_dir, filename)
        embedding = extract_face_embedding(filepath)
        if embedding is not None:
            X.append(embedding)
            y.append(1)

    # Negative samples
    for filename in os.listdir(negative_dir):
        filepath = os.path.join(negative_dir, filename)
        embedding = extract_face_embedding(filepath)
        if embedding is not None:
            X.append(embedding)
            y.append(0)

    if len(X) == 0:
        print("No training data available.")
        return

    X = np.array(X)
    y = np.array(y)

    # Train classifier
    classifier = SVC(kernel='linear', probability=True)
    classifier.fit(X, y)

    # Save classifier
    joblib.dump(classifier, classifier_path)
    print(f"Classifier saved to {classifier_path}")

def classify_faces(faces, classifier_path):
    classifier = joblib.load(classifier_path)

    for face in tqdm(faces, desc="Classifying faces", unit="face"):
        embedding = extract_face_embedding(face.image_path)
        if embedding is not None:
            embedding = embedding.reshape(1, -1)
            probabilities = classifier.predict_proba(embedding)
            label = classifier.predict(embedding)[0]
            confidence = probabilities[0][label]
            face.classification_status = 'success'
            face.classification_label = int(label)  # Convert to native int
            face.classification_confidence = float(confidence)  # Convert to native float
        else:
            face.classification_status = 'failed'
            face.classification_label = None
            face.classification_confidence = None
