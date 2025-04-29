import glob
import os
from video_sequencer.assemble_video import assemble_video
from video_sequencer.generate_image import generate_image
import math

def compute_ramp_endpoints(angle_degrees, canvas_size=(512, 512), margin=64):
    """
    Compute ramp start and end points so ramp fits in canvas with given angle.
    """

    width, height = canvas_size
    angle_rad = math.radians(angle_degrees)

    # How much canvas space is available
    max_dx = width - 2 * margin
    max_dy = height - 2 * margin

    # Based on angle, the effective length must fit within canvas
    # dx = L * cos(angle)
    # dy = L * sin(angle)

    # Maximal ramp length so that dx and dy don't overflow
    length_x_limit = max_dx / math.cos(angle_rad) if math.cos(angle_rad) != 0 else float('inf')
    length_y_limit = max_dy / math.sin(angle_rad) if math.sin(angle_rad) != 0 else float('inf')

    # Choose the smaller limit
    ramp_length_pixels = min(length_x_limit, length_y_limit)

    # Compute dx and dy
    dx = ramp_length_pixels * math.cos(angle_rad)
    dy = ramp_length_pixels * math.sin(angle_rad)

    # Fixed start point (top-left margin)
    ramp_start = (margin, margin)
    ramp_end = (margin + dx, margin + dy)
    
    print(f"ramp_length_pixels: {ramp_length_pixels}")
    print(f"dx: {dx}, dy: {dy}")

    return ramp_start, ramp_end

def generate_frames_and_video(positions, angle_degrees=30.0, ramp_length=5.0, output_folder="frames", fps=5):
    # Clear old frames
    os.makedirs(output_folder, exist_ok=True)
    for file in glob.glob(f"{output_folder}/*.png"):
        os.remove(file)

    ramp_start, ramp_end = compute_ramp_endpoints(angle_degrees)
    
    max_height = ramp_length * math.sin(math.radians(angle_degrees))

    for i, (x_m, y_m) in enumerate(positions):
        generate_image(
            prompt=f"x={x_m:.2f}, y={y_m:.2f}",
            frame_number=i,
            x_m=x_m,
            y_m=y_m,
            ramp_start=ramp_start,
            ramp_end=ramp_end,
            output_folder=output_folder,
            ramp_length=ramp_length,
            max_height=max_height
        )

    output_video_path = os.path.join(output_folder, "output_video.mp4")
    assemble_video(output_folder, output_video_path, fps=fps)
    
    return output_video_path

if __name__ == "__main__":
    positions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    generate_frames_and_video(positions)

