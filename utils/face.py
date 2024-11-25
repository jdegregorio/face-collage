from dataclasses import dataclass, asdict
from typing import Optional

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
    # Classification
    classification_status: str = 'pending'  # 'pending', 'success', 'failed'
    classification_label: Optional[str] = None
    classification_confidence: Optional[float] = None

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(data):
        return Face(**data)
