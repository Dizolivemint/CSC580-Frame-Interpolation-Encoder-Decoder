from PIL import Image, ImageDraw, ImageFont
import os
from utils.normalize_interpolate import normalize_interpolate
from utils.path_utils import resolve_path

def generate_image(prompt, frame_number, normalized_pos, output_folder="frames", ramp_start=(64, 64),ramp_end=(448, 448), img_size=(512, 512), ball_radius=15):
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
    
    # Position of the ball
    x, y = normalize_interpolate(ramp_start, ramp_end, normalized_pos)

    # Draw the ramp line
    d.line([ramp_start, ramp_end], fill=(150, 150, 150), width=3)
    
    # Draw ball
    d.ellipse(
        [(x - ball_radius, y - ball_radius), (x + ball_radius, y + ball_radius)],
        fill=(255, 0, 0), outline=(0, 0, 0)
    )

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    d.text((10, 10), f"Frame {frame_number}: {prompt[:30]}", fill=(0, 0, 0), font=font)
    file_path = resolve_path(f"frame_{frame_number:03}.png", output_folder)
    img.save(file_path)
    return file_path
  
def draw_scene(
    entities,
    frame_number,
    prompt="",
    output_folder="frames",
    img_size=(512, 512),
    background_color=(255, 255, 255)
):
    """
    Draws a general physics frame from a list of entities.

    Args:
        entities (List[Dict]): Each entity has a type, position, shape, etc.
        frame_number (int): Index of the frame
        prompt (str): Annotation label
        output_folder (str): Directory to save frames
        img_size (Tuple[int, int]): Image size
        background_color (Tuple[int, int, int]): RGB color
    """
    os.makedirs(output_folder, exist_ok=True)

    img = Image.new('RGB', img_size, color=background_color)
    draw = ImageDraw.Draw(img)

    for entity in entities:
        etype = entity.get("type")
        pos = entity.get("position")  # (x, y) in pixels
        radius = entity.get("radius", 5)
        color = entity.get("color", (255, 0, 0))

        if etype == "particle":
            x, y = pos
            draw.ellipse(
                [(x - radius, y - radius), (x + radius, y + radius)],
                fill=color, outline=(0, 0, 0)
            )

        elif etype == "line":
            draw.line(entity["points"], fill=color, width=entity.get("width", 2))

        elif etype == "text":
            try:
                font = ImageFont.truetype("arial.ttf", entity.get("size", 16))
            except:
                font = ImageFont.load_default()
            draw.text(pos, entity["text"], fill=color, font=font)

        # Future types: vector fields, heatmaps, force arrows, etc.


    path = resolve_path(f"frame_{frame_number:03}.png", output_folder)
    img.save(path)
    return path