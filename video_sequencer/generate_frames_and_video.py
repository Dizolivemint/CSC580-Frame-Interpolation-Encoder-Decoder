import glob
import os
from video_sequencer.assemble_video import assemble_video
from video_sequencer.generate_image import generate_image

def generate_frames_and_video(positions, output_folder="frames", fps=5):
    # Clear old frames
    os.makedirs(output_folder, exist_ok=True)
    for file in glob.glob(f"{output_folder}/*.png"):
        os.remove(file)

    min_pos = min(positions)
    max_pos = max(positions)

    for i, pos in enumerate(positions):
        normalized = (pos - min_pos) / (max_pos - min_pos + 1e-8)
        generate_image(prompt=f"Pos {pos:.2f}m", frame_number=i, ball_position_normalized=normalized, output_folder=output_folder)

    output_video_path = os.path.join(output_folder, "output_video.mp4")
    assemble_video(output_folder, output_video_path, fps=fps)
    
    return output_video_path

if __name__ == "__main__":
    positions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    generate_frames_and_video(positions)

