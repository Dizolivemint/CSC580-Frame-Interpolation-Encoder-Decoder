# gradio_full_system.py

import gradio as gr
import torch
import pandas as pd
import random
import matplotlib.pyplot as plt
from model_training.model_torch import EncoderDecoder
from model_training.train import train_model
from model_training.generate_training_data import generate_training_data
from video_sequencer.simulate_physics import PhysicsSimulator
from video_sequencer.generate_frames_and_video import generate_frames_and_video
import os
import torch.nn as nn

# --- Available Physics Types ---
physics_types = ["ball_motion", "camera_motion"]  # Extendable easily

# --- Prediction ---
def predict_trajectory(physics_type, *inputs):
    model_path = f"model_training/{physics_type}_model.pth"
    if not os.path.exists(model_path):
        return None

    sample_data = pd.read_pickle(f"data/{physics_type}_data.pkl")
    input_dim = len(sample_data.columns) - 1  # trajectory is 1 column

    output_seq_len = len(sample_data.iloc[0]['trajectory'])
    first_traj = sample_data.iloc[0]['trajectory']
    if isinstance(first_traj[0], (list, tuple)):  # If each timestep is (x,y)
        output_dim = len(first_traj[0])
    else:
        output_dim = 1
        
    model = EncoderDecoder(input_dim=3, output_seq_len=output_seq_len, output_dim=2)
    model.output_layer = nn.Linear(model.decoder_lstm.hidden_size, output_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    inputs_tensor = torch.tensor([inputs], dtype=torch.float32)
    with torch.no_grad():
        prediction = model(inputs_tensor)
        prediction = prediction.cpu().numpy()[0]
        
    return prediction

# --- Plotting ---
def plot_trajectory(physics_type, *inputs):
    pred = predict_trajectory(physics_type, *inputs)
    if pred is None:
        return None
    fig, ax = plt.subplots()
    ax.plot(range(len(pred)), pred, marker='o')
    ax.set_title(f"Predicted Trajectory: {physics_type.replace('_', ' ').title()}")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Position (m)")
    return fig, pred
  
# --- Video Generation ---
def predict_plot_video(physics_type, mass, angle, friction):
    fig, pred = plot_trajectory(physics_type, mass, angle, friction)

    # Generate Frames + Video
    video_mp4_path = generate_frames_and_video(pred, angle_degrees=angle)

    return fig, video_mp4_path

# --- Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("# 🏗️ Full Physics ML System: Data ➔ Train ➔ Predict")

    with gr.Tab("Data Generation"):
        with gr.Row():
            gen_physics_type = gr.Dropdown(choices=physics_types, label="Physics Type")
            num_samples = gr.Slider(100, 5000, value=1000, label="Number of Samples", step=100)
            time_steps = gr.Slider(5, 50, value=10, label="Time Steps", step=5)
        gen_output = gr.Textbox(label="Output Log")
        generate_btn = gr.Button("Generate Data")
        generate_btn.click(
            fn=generate_training_data,
            inputs=[gen_physics_type, num_samples, time_steps],
            outputs=gen_output
        )

    with gr.Tab("Training"):
        with gr.Row():
            train_physics_type = gr.Dropdown(choices=physics_types, label="Physics Type")
            epochs = gr.Slider(5, 100, value=20, label="Epochs", step=1)
        
        early_stopping_checkbox = gr.Checkbox(label="Enable Early Stopping", value=False)
        patience_slider = gr.Slider(1, 20, value=5, label="Patience Steps")
    
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
            inputs=[train_physics_type, epochs, early_stopping_checkbox, patience_slider],
            outputs=[train_output, loss_plot]
        )

    with gr.Tab("Prediction"):
        with gr.Row():
            physics_type = gr.Dropdown(choices=physics_types, label="Physics Type")
            mass = gr.Slider(0.5, 5.0, value=1.0, label="Mass (kg)")
            angle = gr.Slider(5, 45, value=30, label="Ramp Angle (degrees)")
            friction = gr.Slider(0.01, 0.5, value=0.2, label="Friction")
        predict_btn = gr.Button("Predict Trajectory")
        pred_plot = gr.Plot(label="Trajectory Prediction")
        pred_video = gr.Video(label="Generated Video")

        predict_btn.click(
            fn=predict_plot_video,
            inputs=[physics_type, mass, angle, friction],
            outputs=[pred_plot, pred_video]
        )


if __name__ == "__main__":
    demo.launch()
