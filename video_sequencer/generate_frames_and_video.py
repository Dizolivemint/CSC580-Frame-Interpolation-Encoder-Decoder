import glob
import os
from PIL import Image
from video_sequencer.assemble_video import assemble_video
import numpy as np
from utils.path_utils import resolve_path

def generate_frames_and_video(coords, output_folder="frames", fps=5, frame_size=64):
    # Use /tmp directory for frames
    tmp_frames_dir = os.path.join("tmp", output_folder)
    os.makedirs(tmp_frames_dir, exist_ok=True)
    
    # Clean up any existing frames
    for file in glob.glob(os.path.join(tmp_frames_dir, "*.png")):
        os.remove(file)

    for i, (x, y) in enumerate(coords):
        px = int(x * (frame_size - 1))
        py = int(y * (frame_size - 1))
        frame = np.zeros((frame_size, frame_size), dtype=np.uint8)
        frame[py, px] = 255

        img = Image.fromarray(frame).convert("RGB")
        img.save(os.path.join(tmp_frames_dir, f"frame_{i:03}.png"))

    output_video_path = os.path.join(tmp_frames_dir, "output_video.mp4")
    assemble_video(tmp_frames_dir, output_video_path, fps=fps)
    return output_video_path