import random
import pandas as pd
from video_sequencer.simulate_physics import simulate_ball_motion

def generate_training_data(num_samples=1000, time_steps=10, save_path="data/training_data.pkl"):
    samples = []
    for _ in range(num_samples):
        mass = random.uniform(0.5, 5.0)
        angle = random.uniform(10, 45)
        friction = random.uniform(0.05, 0.5)
        trajectory = simulate_ball_motion(mass, angle, friction, time_steps)
        samples.append({
            'mass': mass,
            'angle': angle,
            'friction': friction,
            'trajectory': trajectory
        })
    df = pd.DataFrame(samples)
    df.to_pickle(save_path)
    print(f"\u2705 Saved training data to {save_path}")

if __name__ == "__main__":
    generate_training_data()
