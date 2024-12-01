from dataclasses import dataclass, asdict
from typing import Optional
from datetime import datetime

@dataclass
class Face:
    id: str
    photo_id: str
    image_path: str
    bbox: dict
    actual_expansion: Optional[float] = None
    # Processing status
    head_pose_estimation_status: str = 'pending'  # 'pending', 'success', 'failed'
    head_pose_estimation_error: str = ''
    facial_features_status: str = 'pending'  # 'pending', 'success', 'failed'
    alignment_status: str = 'pending'  # 'pending', 'success', 'failed'
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
    rotation_angle: Optional[float] = None
    scaling_factor: Optional[float] = None
    centering_offsets: Optional[tuple] = None
    # Classification
    classification_status: str = 'pending'  # 'pending', 'success', 'failed'
    classification_label: Optional[int] = None
    classification_confidence: Optional[float] = None
    # Timestamp
    timestamp: Optional[datetime] = None  # Timestamp from the photo

    def to_dict(self):
        data = asdict(self)
        if self.timestamp:
            data['timestamp'] = self.timestamp.isoformat()
        return data

    @staticmethod
    def from_dict(data):
        if 'timestamp' in data and data['timestamp']:
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return Face(**data)
