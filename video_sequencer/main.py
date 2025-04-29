from video_sequencer.simulate_physics import PhysicsSimulator
from prompt_generator import generate_prompts
from video_sequencer.generate_image import generate_fake_image
from video_sequencer.aseemble_video import assemble_video

# 1. Inputs
mass = 1.0
angle = 30.0
friction = 0.2

# 2. Simulate Physics
simulator = PhysicsSimulator()
positions = simulator.simulate_ball_motion(mass, angle, friction, time_steps=5)

# 3. Generate Prompts
prompts = generate_prompts(positions)

# 4. Fake Diffusion - Generate Images
frame_folder = "frames"
for i, prompt in enumerate(prompts):
    generate_fake_image(prompt, i, output_folder=frame_folder)

# 5. Stitch into Video
assemble_video(frame_folder, output_path="output_video.mp4", fps=5)

print("\u2705 Video generated: output_video.mp4")