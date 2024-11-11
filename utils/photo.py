from dataclasses import dataclass, asdict
from typing import Optional
from datetime import datetime, date

@dataclass
class Photo:
    # Metadata from Google Photos
    id: str
    filename: str
    description: Optional[str]
    mimeType: str
    productUrl: str
    baseUrl: str
    creationTime: Optional[str]
    width: Optional[int]
    height: Optional[int]
    photo_cameraMake: Optional[str]
    photo_cameraModel: Optional[str]
    photo_focalLength: Optional[float]
    photo_apertureFNumber: Optional[float]
    photo_isoEquivalent: Optional[int]
    # Date components
    timestamp: Optional[datetime] = None
    year: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None
    weekday: Optional[int] = None
    # Floor dates
    date_floor_day: Optional[date] = None
    date_floor_week: Optional[date] = None
    date_floor_month: Optional[date] = None
    date_floor_year: Optional[date] = None
    # Processing status
    download_status: str = 'pending'  # 'pending', 'success', 'failed'
    download_error: str = ''
    face_detection_status: str = 'pending'  # 'pending', 'success', 'failed'
    face_detection_error: str = ''
    head_pose_estimation_status: str = 'pending'  # 'pending', 'success', 'failed'
    head_pose_estimation_error: str = ''
    facial_features_status: str = 'pending'  # 'pending', 'success', 'failed'
    # Inclusion status
    include_in_collage: bool = True  # Whether to include in the collage
    exclusion_reason: Optional[str] = ''
    # Processing results
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None
    left_eye_openness: Optional[float] = None
    right_eye_openness: Optional[float] = None
    avg_eye_openness: Optional[float] = None
    mouth_openness: Optional[float] = None
    original_image_path: str = ''
    processed_image_path: str = ''
    actual_expansion: Optional[float] = None  # New field

    def to_dict(self):
        data = asdict(self)
        if self.timestamp:
            data['timestamp'] = self.timestamp.isoformat()
        # Serialize date objects
        if self.date_floor_day:
            data['date_floor_day'] = self.date_floor_day.isoformat()
        if self.date_floor_week:
            data['date_floor_week'] = self.date_floor_week.isoformat()
        if self.date_floor_month:
            data['date_floor_month'] = self.date_floor_month.isoformat()
        if self.date_floor_year:
            data['date_floor_year'] = self.date_floor_year.isoformat()
        return data

    @staticmethod
    def from_dict(data):
        if 'timestamp' in data and data['timestamp']:
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if 'date_floor_day' in data and data['date_floor_day']:
            data['date_floor_day'] = datetime.fromisoformat(data['date_floor_day']).date()
        if 'date_floor_week' in data and data['date_floor_week']:
            data['date_floor_week'] = datetime.fromisoformat(data['date_floor_week']).date()
        if 'date_floor_month' in data and data['date_floor_month']:
            data['date_floor_month'] = datetime.fromisoformat(data['date_floor_month']).date()
        if 'date_floor_year' in data and data['date_floor_year']:
            data['date_floor_year'] = datetime.fromisoformat(data['date_floor_year']).date()
        return Photo(**data)
