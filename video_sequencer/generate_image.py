from PIL import Image, ImageDraw, ImageFont
import os

def generate_image(prompt, frame_number, ball_position_normalized, output_folder="frames"):
    """
    Args:
        prompt (str): Text prompt (optional, for annotation)
        frame_number (int): Frame index
        ball_position_normalized (float): 0.0 (top) to 1.0 (bottom) along the ramp
        output_folder (str): Where to save images
    """
    os.makedirs(output_folder, exist_ok=True)
    img_size = (512, 512)
    img = Image.new('RGB', img_size, color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    
    # Ramp: draw a simple diagonal line
    ramp_start = (50, 450)  # bottom-left
    ramp_end = (450, 50)    # top-right
    d.line([ramp_start, ramp_end], fill=(0, 0, 0), width=5)

    # Compute ball position along the ramp
    ball_x = ramp_start[0] + (ramp_end[0] - ramp_start[0]) * ball_position_normalized
    ball_y = ramp_start[1] + (ramp_end[1] - ramp_start[1]) * ball_position_normalized

    # Draw ball
    ball_radius = 15
    d.ellipse(
        [(ball_x - ball_radius, ball_y - ball_radius), (ball_x + ball_radius, ball_y + ball_radius)],
        fill=(255, 0, 0), outline=(0, 0, 0)
    )

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    d.text((10, 10), f"Frame {frame_number}: {prompt[:30]}", fill=(0, 0, 0), font=font)
    file_path = os.path.join(output_folder, f"frame_{frame_number:03}.png")
    img.save(file_path)
    return file_path