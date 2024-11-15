import logging
from simple_term_menu import TerminalMenu
import numpy as np
from tqdm import tqdm
import os
import shutil
from config import PROCESSED_IMAGES_DIR, EXCLUDED_IMAGES_DIR
from datetime import datetime

def exclude_failed_processing_photos(photos):
    total_photos = len(photos)
    failed_photos = [photo for photo in photos if photo.face_detection_status != 'success' or photo.head_pose_estimation_status != 'success' or photo.facial_features_status != 'success']
    failed_count = len(failed_photos)
    failed_percentage = (failed_count / total_photos) * 100 if total_photos > 0 else 0

    print(f"\nTotal photos: {total_photos}")
    print(f"Photos with failed processing: {failed_count} ({failed_percentage:.2f}%)")

    confirmation = input("Do you want to exclude these photos from the collage? (yes/no): ")
    if confirmation.lower() == 'yes':
        for photo in failed_photos:
            if photo.include_in_collage:
                photo.include_in_collage = False
                if photo.exclusion_reason:
                    photo.exclusion_reason += '; Failed processing'
                else:
                    photo.exclusion_reason = 'Failed processing'
        print(f"Excluded {failed_count} photos.")
    else:
        print("No changes made.")

def filter_photos_by_features(photos):
    # Collect features from all photos
    features = {
        'yaw': [],
        'pitch': [],
        'avg_eye_openness': [],
        'mouth_openness': [],
        'actual_expansion': []
    }
    for photo in photos:
        if photo.head_pose_estimation_status == 'success':
            features['yaw'].append(photo.yaw)
            features['pitch'].append(photo.pitch)
            features['avg_eye_openness'].append(photo.avg_eye_openness)
            features['mouth_openness'].append(photo.mouth_openness)
            if photo.actual_expansion is not None:
                features['actual_expansion'].append(photo.actual_expansion)

    # Calculate percentiles
    percentiles = {}
    for feature, values in features.items():
        if values:
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
            print(f"\nPercentiles for {feature} (calculated from all photos):")
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

            # Determine how many photos meet the criteria
            photos_meeting_criteria = [photo for photo in photos if photo.head_pose_estimation_status == 'success' and getattr(photo, feature) is not None and min_value <= getattr(photo, feature) <= max_value]
            total_meeting_criteria = len(photos_meeting_criteria)
            print(f"\nTotal photos meeting the criteria for {feature}: {total_meeting_criteria}")

            # Determine net additional exclusions
            currently_included = [photo for photo in photos if photo.include_in_collage]
            photos_to_exclude = [photo for photo in currently_included if photo.head_pose_estimation_status == 'success' and getattr(photo, feature) is not None and not (min_value <= getattr(photo, feature) <= max_value)]
            net_additional_exclusions = len(photos_to_exclude)
            print(f"Net additional photos to be excluded: {net_additional_exclusions}")

            confirmation = input("Do you want to apply this filter? (yes/no): ")
            if confirmation.lower() == 'yes':
                # Apply filtering
                for photo in photos_to_exclude:
                    photo.include_in_collage = False
                    reason = f"{feature} not in range"
                    if photo.exclusion_reason:
                        photo.exclusion_reason += f'; {reason}'
                    else:
                        photo.exclusion_reason = reason
                print(f"Excluded {net_additional_exclusions} photos based on {feature} filtering.")
            else:
                print("No changes made.")
        else:
            print("Invalid selection. Please try again.")

def filter_photos_by_date(photos):
    # Collect floor dates from all photos
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
            floor_dates = [getattr(photo, f'date_floor_{date_unit}') for photo in photos if getattr(photo, f'date_floor_{date_unit}')]

            if not floor_dates:
                print("No date information available for photos.")
                continue

            min_date = min(floor_dates)
            max_date = max(floor_dates)

            print(f"\nPhoto dates range from {min_date} to {max_date}.")

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

            # Determine how many photos meet the criteria
            photos_meeting_criteria = [photo for photo in photos if getattr(photo, f'date_floor_{date_unit}') and start_date <= getattr(photo, f'date_floor_{date_unit}') <= end_date]
            total_meeting_criteria = len(photos_meeting_criteria)
            print(f"\nTotal photos meeting the date range: {total_meeting_criteria}")

            # Determine net additional exclusions
            currently_included = [photo for photo in photos if photo.include_in_collage]
            photos_to_exclude = [photo for photo in currently_included if not (getattr(photo, f'date_floor_{date_unit}') and start_date <= getattr(photo, f'date_floor_{date_unit}') <= end_date)]
            net_additional_exclusions = len(photos_to_exclude)
            print(f"Net additional photos to be excluded: {net_additional_exclusions}")

            confirmation = input("Do you want to apply this date filter? (yes/no): ")
            if confirmation.lower() == 'yes':
                # Apply filtering
                for photo in photos_to_exclude:
                    photo.include_in_collage = False
                    if photo.exclusion_reason:
                        photo.exclusion_reason += f'; Date not in range ({date_unit})'
                    else:
                        photo.exclusion_reason = f"Date not in range ({date_unit})"
                print(f"Excluded {net_additional_exclusions} photos based on date ({date_unit}) filtering.")
            else:
                print("No changes made.")
        else:
            print("Invalid selection. Please try again.")

def update_status_based_on_file_existence(photos):
    processed_images = set(os.listdir(PROCESSED_IMAGES_DIR))
    excluded_images = set(os.listdir(EXCLUDED_IMAGES_DIR))
    total_changes = 0
    deleted_photos = 0
    moved_photos = 0

    for photo in tqdm(photos, desc="Updating photo statuses", unit="photo"):
        processed_filename = f"{photo.id}.jpg"
        if processed_filename not in processed_images:
            if processed_filename in excluded_images:
                if photo.include_in_collage:
                    photo.include_in_collage = False
                    if photo.exclusion_reason:
                        photo.exclusion_reason += '; Manually excluded'
                    else:
                        photo.exclusion_reason = 'Manually excluded'
                    moved_photos += 1
                    total_changes += 1
            else:
                if photo.include_in_collage:
                    photo.include_in_collage = False
                    if photo.exclusion_reason:
                        photo.exclusion_reason += '; Image file missing'
                    else:
                        photo.exclusion_reason = 'Image file missing'
                    deleted_photos += 1
                    total_changes += 1

    print(f"Updates applied: {total_changes}")
    print(f"Photos moved to excluded_images: {moved_photos}")
    print(f"Photos deleted: {deleted_photos}")

    if total_changes == 0:
        print("No net changes to eligible collage photos.")

def reset_filters(photos):
    confirmation = input("Are you sure you want to reset all filters and include all photos back into the collage? (yes/no): ")
    if confirmation.lower() == 'yes':
        for photo in photos:
            photo.include_in_collage = True
            photo.exclusion_reason = ''
        print("All filters have been reset. All photos are now included in the collage.")
    else:
        print("Reset filters canceled.")

def sample_photos_temporally(photos, total_needed):
    """
    Selects a subset of photos maintaining temporal spacing as much as possible.
    """
    # Sort photos by timestamp
    photos = sorted(photos, key=lambda p: p.timestamp or datetime.now())

    # If total_needed >= len(photos), return all photos
    if total_needed >= len(photos):
        return photos

    # Calculate the interval between selected photos
    interval = len(photos) / total_needed

    selected_photos = []
    for i in range(total_needed):
        index = int(i * interval)
        selected_photos.append(photos[index])

    return selected_photos
