import logging
from simple_term_menu import TerminalMenu
import numpy as np
from tqdm import tqdm
import os
import shutil
from config import PROCESSED_IMAGES_DIR, EXCLUDED_IMAGES_DIR
from datetime import datetime, timedelta

def exclude_failed_processing_photos(photos):
    total_faces = sum(len(photo.face_list) for photo in photos if photo.face_list)
    failed_faces = []
    for photo in photos:
        if photo.face_list:
            for face in photo.face_list:
                if face.head_pose_estimation_status != 'success' or face.facial_features_status != 'success' or face.alignment_status != 'success':
                    failed_faces.append(face)
    failed_count = len(failed_faces)
    failed_percentage = (failed_count / total_faces) * 100 if total_faces > 0 else 0

    print(f"\nTotal faces: {total_faces}")
    print(f"Faces with failed processing: {failed_count} ({failed_percentage:.2f}%)")

    confirmation = input("Do you want to exclude these faces from the collage? (yes/no): ")
    if confirmation.lower() == 'yes':
        for face in failed_faces:
            if face.include_in_collage:
                face.include_in_collage = False
                if face.exclusion_reason:
                    face.exclusion_reason += '; Failed processing'
                else:
                    face.exclusion_reason = 'Failed processing'
        print(f"Excluded {failed_count} faces.")
    else:
        print("No changes made.")

def filter_photos_by_features(photos):
    # Collect features from all faces
    features = {
        'yaw': [],
        'pitch': [],
        'avg_eye_openness': [],
        'mouth_openness': [],
        'actual_expansion': [],
        'rotation_angle': [],
        'scaling_factor': [],
        'centering_offsets': []
    }
    for photo in photos:
        if photo.face_list:
            for face in photo.face_list:
                if face.head_pose_estimation_status == 'success':
                    features['yaw'].append(face.yaw)
                    features['pitch'].append(face.pitch)
                    features['avg_eye_openness'].append(face.avg_eye_openness)
                    features['mouth_openness'].append(face.mouth_openness)
                    if face.actual_expansion is not None:
                        features['actual_expansion'].append(face.actual_expansion)
                    if face.rotation_angle is not None:
                        features['rotation_angle'].append(face.rotation_angle)
                    if face.scaling_factor is not None:
                        features['scaling_factor'].append(face.scaling_factor)
                    if face.centering_offsets is not None:
                        features['centering_offsets'].append(face.centering_offsets)

    # Calculate percentiles
    percentiles = {}
    for feature, values in features.items():
        if feature == 'centering_offsets' and values:
            x_offsets = [offset[0] for offset in values]
            y_offsets = [offset[1] for offset in values]
            percentiles['centering_offsets_x'] = np.percentile(x_offsets, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            percentiles['centering_offsets_y'] = np.percentile(y_offsets, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        elif values:
            percentiles[feature] = np.percentile(values, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        else:
            percentiles[feature] = []

    # Feature selection menu
    options = [
        "1. Yaw",
        "2. Pitch",
        "3. Average Eye Openness",
        "4. Mouth Openness",
        "5. Actual Expansion",
        "6. Return to Previous Menu"
    ]
    terminal_menu = TerminalMenu(options, title="Select a Feature to Filter By")
    while True:
        menu_entry_index = terminal_menu.show()
        if menu_entry_index == 5 or menu_entry_index is None:
            break
        elif menu_entry_index in [0, 1, 2, 3, 4]:
            feature_names = ['yaw', 'pitch', 'avg_eye_openness', 'mouth_openness', 'actual_expansion']
            feature = feature_names[menu_entry_index]
            if len(percentiles[feature]) == 0:
                print(f"No data available for {feature}.")
                continue
            print(f"\nPercentiles for {feature} (calculated from all faces):")
            for i, percentile in enumerate([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]):
                print(f"{percentile}th percentile: {percentiles[feature][i]:.4f}")
            try:
                min_input = input(f"Enter minimum value for {feature} (or press Enter to skip): ")
                min_value = float(min_input) if min_input.strip() != '' else -np.inf
                max_input = input(f"Enter maximum value for {feature} (or press Enter to skip): ")
                max_value = float(max_input) if max_input.strip() != '' else np.inf
            except ValueError:
                print("Invalid input. Please enter numeric values.")
                continue

            # Determine how many faces meet the criteria
            faces_meeting_criteria = []
            for photo in photos:
                if photo.face_list:
                    for face in photo.face_list:
                        if face.head_pose_estimation_status == 'success' and getattr(face, feature) is not None and min_value <= getattr(face, feature) <= max_value:
                            faces_meeting_criteria.append(face)
            total_meeting_criteria = len(faces_meeting_criteria)
            print(f"\nTotal faces meeting the criteria for {feature}: {total_meeting_criteria}")

            # Determine net additional exclusions
            currently_included = []
            for photo in photos:
                if photo.face_list:
                    for face in photo.face_list:
                        if face.include_in_collage:
                            currently_included.append(face)
            faces_to_exclude = []
            for face in currently_included:
                if face.head_pose_estimation_status == 'success' and getattr(face, feature) is not None and not (min_value <= getattr(face, feature) <= max_value):
                    faces_to_exclude.append(face)
            net_additional_exclusions = len(faces_to_exclude)
            print(f"Net additional faces to be excluded: {net_additional_exclusions}")

            confirmation = input("Do you want to apply this filter? (yes/no): ")
            if confirmation.lower() == 'yes':
                # Apply filtering
                for face in faces_to_exclude:
                    face.include_in_collage = False
                    reason = f"{feature} not in range"
                    if face.exclusion_reason:
                        face.exclusion_reason += f'; {reason}'
                    else:
                        face.exclusion_reason = reason
                print(f"Excluded {net_additional_exclusions} faces based on {feature} filtering.")
            else:
                print("No changes made.")
        else:
            print("Invalid selection. Please try again.")

def filter_faces_by_alignment_and_scaling(photos):
    # Collect alignment and scaling data
    features = {
        'rotation_angle': [],
        'scaling_factor': [],
        'actual_expansion': []
    }
    for photo in photos:
        if photo.face_list:
            for face in photo.face_list:
                if face.alignment_status == 'success':
                    if face.rotation_angle is not None:
                        features['rotation_angle'].append(face.rotation_angle)
                    if face.scaling_factor is not None:
                        features['scaling_factor'].append(face.scaling_factor)
                    if face.actual_expansion is not None:
                        features['actual_expansion'].append(face.actual_expansion)

    # Calculate percentiles
    percentiles = {}
    for feature, values in features.items():
        if values:
            percentiles[feature] = np.percentile(values, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        else:
            percentiles[feature] = []

    # Feature selection menu
    options = [
        "1. Rotation Angle",
        "2. Scaling Factor",
        "3. Actual Expansion",
        "4. Return to Previous Menu"
    ]
    terminal_menu = TerminalMenu(options, title="Select Alignment or Scaling Feature to Filter By")
    while True:
        menu_entry_index = terminal_menu.show()
        if menu_entry_index == 3 or menu_entry_index is None:
            break
        elif menu_entry_index in [0, 1, 2]:
            feature_names = ['rotation_angle', 'scaling_factor', 'actual_expansion']
            feature = feature_names[menu_entry_index]
            if len(percentiles[feature]) == 0:
                print(f"No data available for {feature}.")
                continue
            print(f"\nPercentiles for {feature} (calculated from all faces):")
            for i, percentile in enumerate([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]):
                print(f"{percentile}th percentile: {percentiles[feature][i]:.4f}")
            try:
                min_input = input(f"Enter minimum value for {feature} (or press Enter to skip): ")
                min_value = float(min_input) if min_input.strip() != '' else -np.inf
                max_input = input(f"Enter maximum value for {feature} (or press Enter to skip): ")
                max_value = float(max_input) if max_input.strip() != '' else np.inf
            except ValueError:
                print("Invalid input. Please enter numeric values.")
                continue

            # Determine how many faces meet the criteria
            faces_meeting_criteria = []
            for photo in photos:
                if photo.face_list:
                    for face in photo.face_list:
                        if face.alignment_status == 'success' and getattr(face, feature) is not None and min_value <= getattr(face, feature) <= max_value:
                            faces_meeting_criteria.append(face)
            total_meeting_criteria = len(faces_meeting_criteria)
            print(f"\nTotal faces meeting the criteria for {feature}: {total_meeting_criteria}")

            # Determine net additional exclusions
            currently_included = []
            for photo in photos:
                if photo.face_list:
                    for face in photo.face_list:
                        if face.include_in_collage:
                            currently_included.append(face)
            faces_to_exclude = []
            for face in currently_included:
                if face.alignment_status == 'success' and getattr(face, feature) is not None and not (min_value <= getattr(face, feature) <= max_value):
                    faces_to_exclude.append(face)
            net_additional_exclusions = len(faces_to_exclude)
            print(f"Net additional faces to be excluded: {net_additional_exclusions}")

            confirmation = input("Do you want to apply this filter? (yes/no): ")
            if confirmation.lower() == 'yes':
                # Apply filtering
                for face in faces_to_exclude:
                    face.include_in_collage = False
                    reason = f"{feature} not in range"
                    if face.exclusion_reason:
                        face.exclusion_reason += f'; {reason}'
                    else:
                        face.exclusion_reason = reason
                print(f"Excluded {net_additional_exclusions} faces based on {feature} filtering.")
            else:
                print("No changes made.")
        else:
            print("Invalid selection. Please try again.")


def filter_photos_by_date(photos):
    # Collect floor dates from all faces
    date_units = ['day', 'week', 'month', 'year']
    options = [
        "1. Day",
        "2. Week",
        "3. Month",
        "4. Year",
        "5. Return to Previous Menu"
    ]
    terminal_menu = TerminalMenu(options, title="Select Date Unit to Filter By")
    while True:
        menu_entry_index = terminal_menu.show()
        if menu_entry_index == 4 or menu_entry_index is None:
            break
        elif menu_entry_index in [0, 1, 2, 3]:
            date_unit = date_units[menu_entry_index]
            floor_dates = []
            for photo in photos:
                date_value = getattr(photo, f'date_floor_{date_unit}')
                if date_value:
                    if photo.face_list:
                        for face in photo.face_list:
                            floor_dates.append(date_value)

            if not floor_dates:
                print("No date information available for faces.")
                continue

            min_date = min(floor_dates)
            max_date = max(floor_dates)

            print(f"\nFace dates range from {min_date} to {max_date}.")

            try:
                start_date_input = input(f"Enter start date (YYYY-MM-DD) (or press Enter to skip): ")
                if start_date_input.strip() != '':
                    start_date = datetime.strptime(start_date_input.strip(), '%Y-%m-%d').date()
                else:
                    start_date = min_date

                end_date_input = input(f"Enter end date (YYYY-MM-DD) (or press Enter to skip): ")
                if end_date_input.strip() != '':
                    end_date = datetime.strptime(end_date_input.strip(), '%Y-%m-%d').date()
                else:
                    end_date = max_date
            except ValueError:
                print("Invalid date format. Please enter dates in YYYY-MM-DD format.")
                continue

            if start_date > end_date:
                print("Start date must be before or equal to end date.")
                continue

            # Determine how many faces meet the criteria
            faces_meeting_criteria = []
            for photo in photos:
                date_value = getattr(photo, f'date_floor_{date_unit}')
                if date_value and start_date <= date_value <= end_date:
                    if photo.face_list:
                        for face in photo.face_list:
                            faces_meeting_criteria.append(face)
            total_meeting_criteria = len(faces_meeting_criteria)
            print(f"\nTotal faces meeting the date range: {total_meeting_criteria}")

            # Determine net additional exclusions
            currently_included = []
            for photo in photos:
                if photo.face_list:
                    for face in photo.face_list:
                        if face.include_in_collage:
                            currently_included.append(face)
            faces_to_exclude = []
            for face in currently_included:
                date_value = getattr(photo, f'date_floor_{date_unit}')
                if not (date_value and start_date <= date_value <= end_date):
                    faces_to_exclude.append(face)
            net_additional_exclusions = len(faces_to_exclude)
            print(f"Net additional faces to be excluded: {net_additional_exclusions}")

            confirmation = input("Do you want to apply this date filter? (yes/no): ")
            if confirmation.lower() == 'yes':
                # Apply filtering
                for face in faces_to_exclude:
                    face.include_in_collage = False
                    if face.exclusion_reason:
                        face.exclusion_reason += f'; Date not in range ({date_unit})'
                    else:
                        face.exclusion_reason = f"Date not in range ({date_unit})"
                print(f"Excluded {net_additional_exclusions} faces based on date ({date_unit}) filtering.")
            else:
                print("No changes made.")
        else:
            print("Invalid selection. Please try again.")

def filter_photos_by_temporal_clustering(photos):
    # Ask the user for the time gap parameter (in minutes)
    time_gap_input = input("Enter the time gap (in minutes) to define clusters: ")
    try:
        time_gap_minutes = float(time_gap_input)
    except ValueError:
        print("Invalid input. Please enter a numeric value.")
        return

    time_gap = timedelta(minutes=time_gap_minutes)

    # Filter faces that are included in collage and have timestamps
    included_faces = []
    for photo in photos:
        if photo.face_list:
            for face in photo.face_list:
                if face.include_in_collage and photo.timestamp:
                    face.timestamp = photo.timestamp  # Assign photo timestamp to face
                    included_faces.append(face)

    total_included_faces = len(included_faces)

    if not included_faces:
        print("No faces with timestamp information are included in the collage. Cannot perform temporal clustering.")
        return

    # Sort faces by timestamp
    faces_sorted = sorted(included_faces, key=lambda f: f.timestamp)

    # Initialize clusters
    clusters = []
    current_cluster = [faces_sorted[0]]

    for i in range(1, len(faces_sorted)):
        time_diff = faces_sorted[i].timestamp - faces_sorted[i - 1].timestamp
        if time_diff <= time_gap:
            current_cluster.append(faces_sorted[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [faces_sorted[i]]
    # Add the last cluster
    clusters.append(current_cluster)

    # Provide statistics to the user
    total_clusters = len(clusters)
    print(f"\nTotal faces included in collage: {total_included_faces}")
    print(f"Number of clusters detected: {total_clusters}")

    # Determine net additional exclusions
    faces_to_exclude = []
    for cluster in clusters:
        if len(cluster) <= 1:
            continue
        # Find the face with yaw closest to zero
        best_face = None
        min_abs_yaw = None
        for face in cluster:
            if face.head_pose_estimation_status == 'success' and face.yaw is not None:
                abs_yaw = abs(face.yaw)
                if min_abs_yaw is None or abs_yaw < min_abs_yaw:
                    min_abs_yaw = abs_yaw
                    best_face = face

        if best_face is None:
            # If no face has yaw information, select the first one
            best_face = cluster[0]

        # Exclude other faces in the cluster
        for face in cluster:
            if face != best_face and face.include_in_collage:
                faces_to_exclude.append(face)

    net_additional_exclusions = len(faces_to_exclude)
    percentage_excluded = (net_additional_exclusions / total_included_faces) * 100 if total_included_faces > 0 else 0
    print(f"Total faces to be excluded due to temporal clustering: {net_additional_exclusions} ({percentage_excluded:.2f}%)")

    # Ask user for confirmation
    confirmation = input("Do you want to apply this temporal clustering filter? (yes/no): ")
    if confirmation.lower() == 'yes':
        for face in faces_to_exclude:
            face.include_in_collage = False
            reason = f"Excluded due to temporal clustering"
            if face.exclusion_reason:
                face.exclusion_reason += f'; {reason}'
            else:
                face.exclusion_reason = reason
        print(f"Excluded {net_additional_exclusions} faces based on temporal clustering.")
    else:
        print("No changes made.")

def filter_faces_by_classifier(photos):
    total_faces = sum(len(photo.face_list) for photo in photos if photo.face_list)
    classified_faces = []
    for photo in photos:
        if photo.face_list:
            for face in photo.face_list:
                if face.classification_status == 'success' and face.classification_label == 1:
                    classified_faces.append(face)

    total_classified = len(classified_faces)
    print(f"\nTotal faces classified as positive: {total_classified} out of {total_faces}")

    # Calculate confidence percentiles
    confidences = [face.classification_confidence for face in classified_faces if face.classification_confidence is not None]
    if not confidences:
        print("No confidence scores available.")
        return

    percentiles = np.percentile(confidences, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    print("\nConfidence percentiles for positive classifications:")
    for i, percentile in enumerate([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]):
        print(f"{percentile}th percentile: {percentiles[i]:.4f}")

    try:
        min_input = input("Enter minimum confidence value to include (or press Enter to skip): ")
        min_value = float(min_input) if min_input.strip() != '' else 0.0
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return

    # Determine faces to include based on confidence threshold
    faces_to_include = [face for face in classified_faces if face.classification_confidence >= min_value]

    # Update inclusion status
    for photo in photos:
        if photo.face_list:
            for face in photo.face_list:
                if face in faces_to_include:
                    face.include_in_collage = True
                    face.exclusion_reason = ''
                else:
                    face.include_in_collage = False
                    face.exclusion_reason = 'Classifier confidence below threshold'

    included_count = len(faces_to_include)
    print(f"\nTotal faces included after filtering: {included_count}")

def update_status_based_on_file_existence(photos):
    processed_images = set(os.listdir(PROCESSED_IMAGES_DIR))
    excluded_images = set(os.listdir(EXCLUDED_IMAGES_DIR))
    total_changes = 0
    deleted_faces = 0
    moved_faces = 0

    for photo in tqdm(photos, desc="Updating face statuses", unit="photo"):
        if photo.face_list:
            for face in photo.face_list:
                processed_filename = os.path.basename(face.image_path)
                if processed_filename not in processed_images:
                    if processed_filename in excluded_images:
                        if face.include_in_collage:
                            face.include_in_collage = False
                            if face.exclusion_reason:
                                face.exclusion_reason += '; Manually excluded'
                            else:
                                face.exclusion_reason = 'Manually excluded'
                            moved_faces += 1
                            total_changes += 1
                    else:
                        if face.include_in_collage:
                            face.include_in_collage = False
                            if face.exclusion_reason:
                                face.exclusion_reason += '; Image file missing'
                            else:
                                face.exclusion_reason = 'Image file missing'
                            deleted_faces += 1
                            total_changes += 1

    print(f"Updates applied: {total_changes}")
    print(f"Faces moved to excluded_images: {moved_faces}")
    print(f"Faces deleted: {deleted_faces}")

    if total_changes == 0:
        print("No net changes to eligible collage faces.")

def reset_filters(photos):
    confirmation = input("Are you sure you want to reset all filters and include all faces back into the collage? (yes/no): ")
    if confirmation.lower() == 'yes':
        for photo in photos:
            if photo.face_list:
                for face in photo.face_list:
                    face.include_in_collage = True
                    face.exclusion_reason = ''
        print("All filters have been reset. All faces are now included in the collage.")
    else:
        print("Reset filters canceled.")

def sample_photos_temporally(faces, total_needed):
    """
    Selects a subset of faces maintaining temporal spacing as much as possible.
    """
    from datetime import datetime

    # Filter faces with valid timestamps
    faces_with_timestamp = [face for face in faces if face.timestamp is not None]

    if not faces_with_timestamp:
        print("No faces have timestamp information. Cannot sample temporally.")
        return faces[:total_needed]  # Return first N faces

    # Sort faces by timestamp
    faces_sorted = sorted(faces_with_timestamp, key=lambda f: f.timestamp)

    # If total_needed >= len(faces), return all faces
    if total_needed >= len(faces_sorted):
        return faces_sorted

    # Extract timestamps as floats for computation
    timestamps = [f.timestamp.timestamp() for f in faces_sorted]
    min_timestamp = timestamps[0]
    max_timestamp = timestamps[-1]

    selected_indices = set()
    selected_faces = []

    for i in range(total_needed):
        # Compute the target timestamp using percentiles
        if total_needed > 1:
            percentile = i / (total_needed - 1)  # Percentile between 0 and 1
        else:
            percentile = 0.5  # If only one face is needed, pick the middle one
        target_timestamp = min_timestamp + percentile * (max_timestamp - min_timestamp)

        # Find the closest face to the target timestamp that's not already selected
        closest_index = min(
            (j for j in range(len(timestamps)) if j not in selected_indices),
            key=lambda j: abs(timestamps[j] - target_timestamp)
        )
        selected_indices.add(closest_index)
        selected_faces.append(faces_sorted[closest_index])

    # Sort the selected faces by timestamp (optional)
    selected_faces.sort(key=lambda f: f.timestamp)

    return selected_faces
