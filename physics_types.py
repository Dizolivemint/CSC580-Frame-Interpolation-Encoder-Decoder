from typing import TypedDict, Literal, Union, List, Tuple
import numpy as np

# Define supported physics types
PhysicsType = Literal["ball_motion", "camera_motion"]

# Common types for reusable use
Frame = np.ndarray  # shape: (H, W), dtype=float32
Trajectory = List[Frame]  # list of (H, W) frames

class BallMotionSample(TypedDict):
    mass: float
    angle: float
    friction: float
    trajectory: Trajectory

class CameraMotionSample(TypedDict):
    initial_velocity: float
    acceleration: float
    trajectory: Trajectory

# Union type for any simulation sample
PhysicsSample = Union[BallMotionSample, CameraMotionSample]
