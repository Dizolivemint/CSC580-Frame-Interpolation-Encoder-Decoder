from moviepy import ImageSequenceClip
import os

def assemble_video(frame_folder, output_mp4_path, output_webm_path=None, fps=10):
    images = sorted([
        os.path.join(frame_folder, img)
        for img in os.listdir(frame_folder)
        if img.endswith(".png")
    ])
    
    clip = ImageSequenceClip(images, fps=fps)

    # Save MP4 (H.264)
    clip.write_videofile(output_mp4_path, codec='libx264', audio=False)

    if output_webm_path:
        # Save WebM (VP8 or VP9)
        clip.write_videofile(output_webm_path, codec='vp9', audio=False)