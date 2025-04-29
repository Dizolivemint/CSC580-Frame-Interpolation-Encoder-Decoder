from PIL import Image, ImageDraw, ImageFont
import os

def generate_fake_image(prompt, frame_number, output_folder="frames"):
    os.makedirs(output_folder, exist_ok=True)
    img = Image.new('RGB', (512, 512), color = (255, 255, 255))
    d = ImageDraw.Draw(img)
    
    # Attempt to load a font or default to system font
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    d.text((10,10), prompt, fill=(0,0,0), font=font)
    file_path = os.path.join(output_folder, f"frame_{frame_number:03}.png")
    img.save(file_path)
    return file_path