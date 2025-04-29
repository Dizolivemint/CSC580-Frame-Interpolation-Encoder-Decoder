from PIL import Image, ImageDraw, ImageFont
import os

def generate_image(prompt, frame_number, x_m, y_m, output_folder="frames", ramp_start = (450, 450),ramp_end = (50, 50), ramp_length=5.0, max_height=2.5):
    """
    Args:
        prompt (str): Text prompt (optional, for annotation)
        frame_number (int): Frame index
        ball_position_normalized (float): 0.0 (top) to 1.0 (bottom) along the ramp
        output_folder (str): Where to save images
        ramp_start: start of ramp in pixel coords
        ramp_end: end of ramp in pixel coords
    """
    os.makedirs(output_folder, exist_ok=True)
    img_size = (512, 512)
    img = Image.new('RGB', img_size, color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    
    d.line([ramp_start, ramp_end], fill=(0, 0, 0), width=5)

    # Convert real-world (x, y) to canvas coordinates
    # We'll map x in [0, ramp_length] → pixels between ramp_start[0] to ramp_end[0]
    # and y in [0, max_height] → ramp_start[1] to ramp_end[1] (inverted y-axis)
    x_ratio = x_m / ramp_length
    y_ratio = y_m / max_height

    ball_x = ramp_start[0] + (ramp_end[0] - ramp_start[0]) * x_ratio
    ball_y = ramp_start[1] + (ramp_end[1] - ramp_start[1]) * y_ratio

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