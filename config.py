# config.py

from typing import List
from video_sequencer.simulate_physics import PhysicsSimulator
import os

# --- Input Normalization Configs ---
physics_input_configs = {
    "ball_motion": {
        "fields": ["mass", "angle", "friction"],
        "normalize": lambda m, a, f: [
            m / 5.0,
            (a - 10.0) / 35.0,
            (f - 0.05) / 0.45
        ],
        "denormalize": lambda x: (
            x[0] * 5.0,
            x[1] * 35.0 + 10.0,
            x[2] * 0.45 + 0.05
        )
    },
    "projectile_motion": {
        "fields": ["initial_velocity", "angle", "gravity"],
        "normalize": lambda v0, a, g: [
            v0 / 20.0,
            a / 90.0,
            g / 20.0
        ],
        "denormalize": lambda x: (
            x[0] * 20.0,
            x[1] * 90.0,
            x[2] * 20.0
        )
    }
}

# --- Shared Accessor Functions ---
def get_physics_types() -> List[str]:
    return list(physics_input_configs.keys())

def normalize_input(physics_type: str, *inputs: float) -> List[float]:
    if physics_type not in physics_input_configs:
        raise ValueError(f"Unsupported physics type: {physics_type}")
    return physics_input_configs[physics_type]["normalize"](*inputs)

def denormalize_input(physics_type: str, inputs: List[float]) -> List[float]:
    if physics_type not in physics_input_configs:
        raise ValueError(f"Unsupported physics type: {physics_type}")
    return physics_input_configs[physics_type]["denormalize"](inputs)

def get_input_fields(physics_type: str) -> List[str]:
    if physics_type not in physics_input_configs:
        raise ValueError(f"Unsupported physics type: {physics_type}")
    return physics_input_configs[physics_type]["fields"]

def get_simulation_fn(physics_type):
    sim = PhysicsSimulator()
    return {
        "ball_motion": sim.simulate_ball_motion,
        "projectile_motion": sim.simulate_projectile_motion
    }.get(physics_type)

def get_param_ranges(physics_type):
    if physics_type == "ball_motion":
        return {
            "mass": (0.5, 5.0),
            "angle": (10.0, 45.0),
            "friction": (0.05, 0.5),
        }
    elif physics_type == "projectile_motion":
        return {
            "initial_velocity": (25.0, 50.0),
            "angle": (45.0, 75.0),
            "gravity": (9.5, 10.5),
        }
    else:
        raise ValueError(f"Unknown physics type: {physics_type}")