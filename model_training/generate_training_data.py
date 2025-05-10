import random
import pandas as pd
import numpy as np
import time
from config import get_input_fields, get_simulation_fn, normalize_input, get_param_ranges
from utils.path_utils import resolve_path

def generate_training_data(physics_type, num_samples=1000, time_steps=50):
    samples = []
    
    start_total = time.time()
    slowest = 0.0
    worst_case = (0.0, 0.0, 0.0)  # Initialize with default values
    skipped = 0
    
    param_specs = get_input_fields(physics_type)
    param_ranges = get_param_ranges(physics_type)
    sim_fn = get_simulation_fn(physics_type)
    
    for i in range(num_samples):
        t0 = time.time()
        
        # --- Randomize parameters based on config ---
        param_values = {
            field: random.uniform(*param_ranges[field])
            for field in param_specs
        }

        try:
            trajectory = sim_fn(**param_values, time_steps=time_steps)
        except Exception as e:
            print(f"⚠️ Skipping sample due to error: {e}")
            skipped += 1
            continue
        
        dot_coords = []
        for frame in trajectory:
            coords = np.argwhere(frame > 0.5)
            if coords.size:
                y, x = coords[0]
                x_norm, y_norm = x / 63.0, y / 63.0
                dot_coords.append((x_norm, y_norm))
            else:
                dot_coords.append((0.0, 0.0))
                
        # --- Debug ---
        # if i < 3:
        #     print(f"Sample {i} raw active coords:")
        #     for t, frame in enumerate(trajectory[:10]):
        #         coords = np.argwhere(frame > 0.5)
        #         if coords.size:
        #             print(f"  t={t} dot at {coords[0].tolist()}")
        #         else:
        #             print(f"  t={t} no dot")


        # Skip if ball stays mostly in one place
        unique_coords = set(dot_coords)

        if len(unique_coords) < 3:
            if i < 5:
                print(f"Skipping sample {i} due to mostly stationary dot")
            skipped += 1
            continue
          
        # Collect values for metadata
        row_data = {field: param_values[field] for field in param_specs}
        row_data["trajectory"] = dot_coords
        samples.append(row_data)
        
        # --- Track performance ---
        t1 = time.time()
        dt = t1 - t0
        if dt > slowest:
            slowest = dt
            worst_case = param_values
        
        # Debug
        # if i % 100 == 0:
        #     print(f"Sample {i}, time={dt:.4f}s")

    df = pd.DataFrame(samples)
    filename = resolve_path(f"{physics_type}_data.pkl", write_mode=True)
    df.to_pickle(filename)
    print(f"⏱️ Total time: {time.time() - start_total:.2f}s")
    if len(samples) > 0:
        print(f"⏱️ Worst-case time per sample: {slowest:.4f}s → " + 
              ", ".join(f"{k}={v:.2f}" for k, v in worst_case.items() if k != "trajectory"))
    print(f"✅ Saved {len(samples)} samples to {filename} (skipped {skipped} stationary ones)")
    return f"✅ Saved {len(samples)} samples to {filename}"

if __name__ == "__main__":
    generate_training_data()
