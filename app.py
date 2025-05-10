# gradio_full_system.py

import gradio as gr
import torch
import pandas as pd
import random
import matplotlib.pyplot as plt
from model_training.model_torch import EncoderDecoder
from model_training.train import train_model
from model_training.generate_training_data import generate_training_data
from model_training.train import PhysicsTrajectoryDataset
from video_sequencer.generate_frames_and_video import generate_frames_and_video
import os
import torch.nn as nn
import numpy as np
from config import normalize_input, denormalize_input, get_input_fields, get_physics_types, get_param_ranges
from utils.path_utils import resolve_path
from video_sequencer.simulate_physics import PhysicsSimulator

# --- Available Physics Types ---
physics_types = get_physics_types()

# --- Inspect Training Trajectories ---
def inspect_training_trajectories(physics_type, frame_size=64):
    dataset_path = resolve_path(f"{physics_type}_data.pkl")
    dataset = PhysicsTrajectoryDataset(dataset_path, physics_type)

    num_samples = 3
    fig, axes = plt.subplots(num_samples, 3, figsize=(10, 3 * num_samples), constrained_layout=True)

    for i in range(num_samples):
        row = dataset.df.iloc[i]
        input_features, trajectory = dataset[i]  # trajectory: [T, 2]
        T = trajectory.shape[0]
        angle = row.get("angle", None)

        for j, t in enumerate([0, T // 2, T - 1]):
            ax = axes[i, j] if num_samples > 1 else axes[j]
            frame = torch.zeros((frame_size, frame_size))

            # Get (x, y) and denormalize back to pixel space
            x, y = trajectory[t].numpy()
            px = int(x * (frame_size - 1))
            py = int(y * (frame_size - 1))

            # Plot the point
            ax.imshow(frame, cmap="gray")
            ax.plot(px, py, "ro", markersize=5)

            # Initialize title
            title = f"t={t}"

            # Dynamically append available metadata fields
            for key in ['mass', 'angle', 'friction', 'initial_velocity', 'acceleration', 'gravity']:
                if key in row:
                    val = row[key]
                    if key == 'angle':
                        title += f"\n{key.capitalize()}: {val:.1f}°"
                    else:
                        title += f"\n{key.capitalize()}: {val:.2f}"
            if angle is not None:
                title += f"\nAngle: {angle:.1f}°"

            ax.set_title(title, pad=10)
            ax.axis("off")

    return fig
  
# --- Predict Normalized Coordinates ---
def predict_trajectory(physics_type, *inputs, debug=False):
    model_file = f"{physics_type}_model.pth"
    model_path = resolve_path(model_file)
    if not os.path.exists(model_path):
        return None
    
    sample_data = pd.read_pickle(resolve_path(f"{physics_type}_data.pkl"))
    input_dim = len(sample_data.columns) - 1
    output_seq_len = len(sample_data.iloc[0]['trajectory'])  # [T, 2]

    model = EncoderDecoder(
        input_dim=input_dim,
        output_seq_len=output_seq_len,
        output_shape=None  # Use coordinate mode
    )
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    inputs_tensor = torch.tensor([inputs], dtype=torch.float32)
    with torch.no_grad():
        prediction = model(inputs_tensor)
        prediction = prediction.cpu().numpy()[0]  # [T, 2]

        if debug:
            print("🔍 Debug output — predicted (x, y) per frame:")
            for t, (x, y) in enumerate(prediction):
                print(f"t={t}: ({x:.3f}, {y:.3f})")

    return prediction  # [T, 2]

# --- Plotting for Normalized (x, y) ---
def plot_coordinates_over_time(coords, title="Predicted Dot Trajectory", frame_size=64):
    T = len(coords)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    indices = [0, T // 2, T - 1]

    for ax, idx in zip(axes, indices):
        x, y = coords[idx]
        px = int(x * (frame_size - 1))
        py = int(y * (frame_size - 1))

        frame = np.zeros((frame_size, frame_size))
        ax.imshow(frame, cmap='gray')
        ax.plot(px, py, 'ro')
        ax.set_title(f"t={idx} | ({px}, {py})")
        ax.axis('off')

    fig.suptitle(title)
    return fig

def plot_multiple_predictions(predictions, inputs_list, frame_size=64, denorm=None):
    N = len(predictions)
    fig, axes = plt.subplots(N, 3, figsize=(12, 3 * N), constrained_layout=True)

    for row in range(N):
        indices = [0, len(predictions[row]) // 2, len(predictions[row]) - 1]
        inp = inputs_list[row]

        # Denormalize if provided
        labels = denorm(inp) if denorm else inp

        for col, t in enumerate(indices):
            x, y = predictions[row][t]
            px = int(x * (frame_size - 1))
            py = int(y * (frame_size - 1))
            
            px = round(px, 2)
            py = round(py, 2)

            frame = np.zeros((frame_size, frame_size))
            ax = axes[row, col] if N > 1 else axes[col]
            ax.imshow(frame, cmap='gray')
            ax.plot(px, py, 'ro')
            ax.set_title(f"t={t} | inputs={labels}\nDot: ({px}, {py})")
            ax.axis("off")

    return fig
  
def plot_trajectory(physics_type, *inputs, debug=False):
    pred = predict_trajectory(physics_type, *inputs, debug=debug)
    if pred is None:
        return None
    fig = plot_coordinates_over_time(pred, title=f"{physics_type.title()} Prediction")
    return fig, pred

def extract_xy_from_frame_sequence(frames):
    coords = []
    for frame in frames:
        yx = np.argwhere(frame == 1.0)
        if len(yx) > 0:
            y, x = yx[0]
            coords.append([x / 63.0, 1.0 - y / 63.0])  # match your prediction's normalized format
        else:
            coords.append([0.0, 0.0])  # fallback
    return np.array(coords)  # [T, 2]

# --- Video Generation ---
def predict_plot_video(physics_type, *inputs, debug=False):
    norm_input = normalize_input(physics_type, *inputs)

    pred_fig, pred_coords = plot_trajectory(physics_type, *norm_input, debug=debug)
    if pred_coords is None:
        return None, None, None
      
    dataset = PhysicsTrajectoryDataset(resolve_path(f"{physics_type}_data.pkl"), physics_type)

    # Search for closest match using the normalized inputs already produced
    closest_idx = None
    closest_dist = float('inf')

    for i in range(len(dataset)):
        dataset_input, dataset_trajectory = dataset[i]
        dist = np.linalg.norm(dataset_input.numpy() - np.array(norm_input))
        if dist < closest_dist:
            closest_dist = dist
            closest_idx = i

    actual_coords = dataset[closest_idx][1].numpy()

    actual_fig = plot_coordinates_over_time(actual_coords, title=f"{physics_type.title()} Actual")
    
    video_mp4_path = generate_frames_and_video(pred_coords)
    return pred_fig, actual_fig, video_mp4_path

def test_input_sensitivity(physics_type):
    model_path = resolve_path(f"{physics_type}_model.pth")
    if not os.path.exists(model_path):
        return None

    # Load sample data to get dimensions
    sample_data = pd.read_pickle(f"data/{physics_type}_data.pkl")
    input_dim = len(sample_data.columns) - 1
    output_seq_len = len(sample_data.iloc[0]['trajectory'])  # [T, 2]

    # Load trained model
    model = EncoderDecoder(
        input_dim=input_dim,
        output_seq_len=output_seq_len,
        output_shape=None
    )
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    # Get parameter ranges and field names
    param_ranges = get_param_ranges(physics_type)
    param_specs = get_input_fields(physics_type)
    
    # Dynamically generate test inputs (midpoints or meaningful values)
    test_inputs = []
    for variation in range(4):
        base = []
        for field in param_specs:
            min_val, max_val = param_ranges[field]
            mid = (min_val + max_val) / 2
            val = mid + ((variation - 1.5) * (max_val - min_val) / 4)
            val = max(min_val, min(max_val, val))  # Clamp
            base.append(val)
        test_inputs.append(normalize_input(physics_type, *base))

    preds = []
    for inp in test_inputs:
        inp_tensor = torch.tensor([inp], dtype=torch.float32)
        with torch.no_grad():
            output = model(inp_tensor).cpu().numpy()[0]  # [T, 2]
        preds.append(output)

    # Create a closure that captures physics_type
    def denorm_fn(inputs):
        return denormalize_input(physics_type, inputs)

    fig = plot_multiple_predictions(preds, test_inputs, denorm=denorm_fn)
    return fig
  
def load_uploaded_dataset(uploaded_file, physics_type):
    tmp_path = os.path.join("/tmp", f"{physics_type}_data.pkl")
    with open(uploaded_file.name, "rb") as src, open(tmp_path, "wb") as dst:
        dst.write(src.read())
    return f"✅ Uploaded and saved dataset to /tmp for '{physics_type}'"

def load_uploaded_model(uploaded_file, physics_type):
    tmp_path = os.path.join("/tmp", "", f"{physics_type}_model.pth")
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    with open(uploaded_file.name, "rb") as src, open(tmp_path, "wb") as dst:
        dst.write(src.read())
    return f"✅ Uploaded and saved model to /tmp for '{physics_type}'"

# --- Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Full Physics ML System")
    gr.Markdown("## 🧪 Data ➔ Train ➔ Predict")
    gr.Markdown("### 👨‍💻 Developed by [Miles Exner](https://www.linkedin.com/in/milesexner/)")

    # 🔄 Global dropdown visible to user
    physics_dropdown = gr.Dropdown(
        choices=physics_types,
        label="Physics Type",
        value=physics_types[0]
    )

    with gr.Tab("Data Generation"):
        with gr.Row():
            num_samples = gr.Slider(100, 5000, value=1000, label="Number of Samples", step=100)
            time_steps = gr.Slider(5, 100, value=50, label="Time Steps", step=5)
        gen_output = gr.Textbox(label="Output Log")
        generate_btn = gr.Button("Generate Data")
        generate_btn.click(
            fn=generate_training_data,
            inputs=[physics_dropdown, num_samples, time_steps],
            outputs=gen_output
        )
        
        gr.Markdown("#### 📤 Upload Existing Dataset (.pkl)")
        upload_data = gr.File(file_types=[".pkl"], label="Upload .pkl")
        upload_data.upload(
            fn=load_uploaded_dataset,
            inputs=[upload_data, physics_dropdown],
            outputs=gen_output
        )

        gr.Markdown("#### 📥 Download Generated Dataset")
        download_data_btn = gr.Button("Download Dataset")
        download_data_file = gr.File(label="Download Link")

        def return_dataset_path(physics_type):
            return resolve_path(f"{physics_type}_data.pkl", write_mode=True)

        download_data_btn.click(
            fn=return_dataset_path,
            inputs=[physics_dropdown],
            outputs=download_data_file
        )

    
    with gr.Tab("Data Inspection"):
        gr.Markdown("Visualize 3 samples from the training dataset to debug dot position.")

        inspect_btn = gr.Button("Show Sample Trajectories")
        output_fig = gr.Plot()

        inspect_btn.click(
            fn=inspect_training_trajectories,
            inputs=[physics_dropdown],
            outputs=output_fig
        )

    with gr.Tab("Training"):
        with gr.Row():
            epochs = gr.Slider(5, 100, value=20, label="Epochs", step=1)
        
        early_stopping_checkbox = gr.Checkbox(label="Enable Early Stopping", value=False)
        patience_slider = gr.Slider(1, 20, value=5, label="Patience Steps", step=1)
    
        train_output = gr.Textbox(label="Training Log")
        loss_plot = gr.Plot(label="Training Loss Curve")
        train_btn = gr.Button("Train Model")

        def run_training(physics_type, epochs, early_stopping, patience):
            msg, losses = train_model(
                physics_type=physics_type,
                epochs=epochs,
                early_stopping=early_stopping,
                patience=patience
            )
            fig, ax = plt.subplots()
            ax.plot(losses)
            ax.set_title("Training Loss Curve")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            return msg, fig

        train_btn.click(
            fn=run_training,
            inputs=[physics_dropdown, epochs, early_stopping_checkbox, patience_slider],
            outputs=[train_output, loss_plot]
        )
        
        gr.Markdown("#### 📤 Upload Trained Model (.pth)")
        upload_model = gr.File(file_types=[".pth"], label="Upload .pth")
        upload_model.upload(
            fn=load_uploaded_model,
            inputs=[upload_model, physics_dropdown],
            outputs=train_output
        )

        gr.Markdown("#### 📥 Download Trained Model")
        download_model_btn = gr.Button("Download Model")
        download_model_file = gr.File(label="Download Link")

        def return_model_path(physics_type):
            return resolve_path(f"{physics_type}_model.pth", write_mode=True)

        download_model_btn.click(
            fn=return_model_path,
            inputs=[physics_dropdown],
            outputs=download_model_file
        )

        
    with gr.Tab("Input Sensitivity Test"):
        gr.Markdown("Compare how different inputs affect predicted trajectories.")
        sensitivity_btn = gr.Button("Run Sensitivity Test")
        output_plot = gr.Plot()

        sensitivity_btn.click(
            fn=test_input_sensitivity,
            inputs=[physics_dropdown],
            outputs=output_plot
      )

    with gr.Tab("Prediction"):
        with gr.Row():
            slider_outputs = [
              gr.Slider(visible=False),
              gr.Slider(visible=False),
              gr.Slider(visible=False)
            ]
            
        debug_checkbox = gr.Checkbox(label="Debug", value=False)
        predict_btn = gr.Button("Predict Trajectory")
        pred_plot = gr.Plot(label="Predicted Trajectory")
        actual_plot = gr.Plot(label="Actual Physics Trajectory")
        pred_video = gr.Video(label="Generated Video")

        def refresh_inputs(physics_type):
            sliders = get_input_fields(physics_type)  # Returns configured sliders
            updates = []
            param_ranges = get_param_ranges(physics_type)  
            for template, target in zip(sliders, slider_outputs):
                min_val, max_val = param_ranges[template]
                default_val = (min_val + max_val) / 2
                updates.append(gr.update(
                    visible=True,
                    label=template.replace('_', ' ').title(),
                    minimum=min_val,
                    maximum=max_val,
                    value=default_val
                ))
                
            # Hide any unused sliders
            for _ in range(len(sliders), len(slider_outputs)):
                updates.append(gr.update(visible=False))
                
            return updates

        physics_dropdown.change(
            fn=refresh_inputs,
            inputs=physics_dropdown,
            outputs=slider_outputs
        )
        
        demo.load(fn=refresh_inputs, inputs=physics_dropdown, outputs=slider_outputs)

        def predict_switch(physics_type, *args):
            *slider_vals, debug = args
            return predict_plot_video(physics_type, *slider_vals, debug=debug)

        predict_btn.click(
            fn=predict_switch,
            inputs=[physics_dropdown] + slider_outputs + [debug_checkbox],
            outputs=[pred_plot, actual_plot, pred_video]
        )
            
if __name__ == "__main__":
    demo.launch()
