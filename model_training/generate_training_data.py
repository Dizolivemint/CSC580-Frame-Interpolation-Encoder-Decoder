import random
import pandas as pd
from video_sequencer.simulate_physics import PhysicsSimulator

def generate_training_data(physics_type, num_samples=1000, time_steps=50, save_dir="data/"):
    simulator = PhysicsSimulator()
    samples = []
    
    for _ in range(num_samples):
        if physics_type == "ball_motion":
            mass = random.uniform(0.5, 5.0)
            angle = random.uniform(10, 45)
            friction = random.uniform(0.05, 0.5)
            trajectory = simulator.simulate_ball_motion(mass, angle, friction, time_steps)
            samples.append({
                'mass': mass,
                'angle': angle,
                'friction': friction,
                'trajectory': trajectory
            })
        
        elif physics_type == "camera_motion":
            initial_velocity = random.uniform(0, 5)
            acceleration = random.uniform(-1, 1)
            trajectory = simulator.simulate_camera_motion(initial_velocity, acceleration, time_steps)
            samples.append({
                'initial_velocity': initial_velocity,
                'acceleration': acceleration,
                'trajectory': trajectory
            })
        
        else:
            raise ValueError(f"Unknown physics type: {physics_type}")

    df = pd.DataFrame(samples)
    filename = f"{save_dir}/{physics_type}_data.pkl"
    df.to_pickle(filename)
    print(f"✅ Saved {num_samples} samples to {filename}")
    return f"✅ Saved {num_samples} samples to {filename}"

if __name__ == "__main__":
    generate_training_data()
