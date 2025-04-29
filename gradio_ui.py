# gradio_full_system.py

import gradio as gr
import torch
import pandas as pd
import random
import matplotlib.pyplot as plt
from model_training.model_torch import EncoderDecoder
from video_sequencer.simulate_physics import PhysicsSimulator
import os

# --- Available Physics Types ---
physics_types = ["ball_motion", "camera_motion"]  # Extendable easily

# --- Data Generation ---
def generate_training_data(physics_type, num_samples=1000, time_steps=10, save_dir="data/"):
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

    df = pd.DataFrame(samples)
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{save_dir}/{physics_type}_data.pkl"
    df.to_pickle(filename)
    return f"✅ Saved {num_samples} samples to {filename}"

# --- Training ---
def train_model(physics_type, hidden_size=64, lr=0.001, epochs=20):
    data_path = f"data/{physics_type}_data.pkl"
    model_path = f"model_training/{physics_type}_model.pth"

    df = pd.read_pickle(data_path)
    if physics_type == "ball_motion":
        X = torch.tensor(df[['mass', 'angle', 'friction']].values, dtype=torch.float32)
    elif physics_type == "camera_motion":
        X = torch.tensor(df[['initial_velocity', 'acceleration']].values, dtype=torch.float32)
    else:
        raise ValueError(f"Unknown physics type: {physics_type}")

    y = torch.tensor(df['trajectory'].tolist(), dtype=torch.float32)
    model = EncoderDecoder(output_seq_len=y.shape[1])
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    torch.save(model.state_dict(), model_path)
    return f"✅ Trained and saved {physics_type} model to {model_path}", losses

# --- Prediction ---
def predict_trajectory(physics_type, *inputs):
    model_path = f"model_training/{physics_type}_model.pth"
    if not os.path.exists(model_path):
        return None

    sample_data = pd.read_pickle(f"data/{physics_type}_data.pkl")
    input_dim = len(sample_data.columns) - 1  # trajectory is 1 column

    output_seq_len = len(sample_data.iloc[0]['trajectory'])
    model = EncoderDecoder(output_seq_len=output_seq_len)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    inputs_tensor = torch.tensor([inputs], dtype=torch.float32)
    with torch.no_grad():
        prediction = model(inputs_tensor).numpy()[0]
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
    return fig

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
        train_output = gr.Textbox(label="Training Log")
        loss_plot = gr.Plot(label="Training Loss Curve")
        train_btn = gr.Button("Train Model")

        def run_training(physics_type, epochs):
            msg, losses = train_model(physics_type, epochs=epochs)
            fig, ax = plt.subplots()
            ax.plot(losses)
            ax.set_title("Training Loss Curve")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            return msg, fig

        train_btn.click(
            fn=run_training,
            inputs=[train_physics_type, epochs],
            outputs=[train_output, loss_plot]
        )

    with gr.Tab("Prediction"):
        with gr.Row():
            pred_physics_type = gr.Dropdown(choices=physics_types, label="Physics Type")
            mass = gr.Slider(0.5, 5.0, value=1.0, label="Mass (kg)")
            angle = gr.Slider(5, 45, value=30, label="Ramp Angle (degrees)")
            friction = gr.Slider(0.01, 0.5, value=0.2, label="Friction")
        predict_btn = gr.Button("Predict Trajectory")
        pred_plot = gr.Plot(label="Trajectory Prediction")

        predict_btn.click(
            fn=plot_trajectory,
            inputs=[pred_physics_type, mass, angle, friction],
            outputs=pred_plot
        )

if __name__ == "__main__":
    demo.launch()
