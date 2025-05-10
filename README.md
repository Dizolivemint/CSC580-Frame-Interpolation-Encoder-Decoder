# 🧠 Motion Encoder Decoder ML Pipeline

An interactive Gradio-based machine learning pipeline for generating, training, and testing encoder-decoder models on simulated physics trajectories.

This PyTorch-based system models motion dynamics such as projectile paths and bouncing objects using a sequence-to-sequence architecture.

## Features

- ⚙️ Dataset generation (custom physics simulations)
- 🧪 Training with optional early stopping
- 📈 Input sensitivity testing
- 🔮 Real-time predictions and trajectory visualizations
- 📤 Upload / 📥 Download of models and datasets (in `/tmp`)

## Try It Out

1. Select a physics type
2. Generate or upload a dataset
3. Train a model or upload a pretrained `.pth`
4. Visualize predictions from dynamic input sliders

## Built With

- Python 3.13
- PyTorch
- Gradio
- Matplotlib
- NumPy

---

👨‍💻 Developed by [Miles Exner](https://www.linkedin.com/in/milesexner/)
