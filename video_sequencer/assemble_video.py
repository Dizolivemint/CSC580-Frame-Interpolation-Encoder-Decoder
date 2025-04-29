import cv2
import os

def assemble_video(frame_folder, output_path, fps=10):
    images = [img for img in sorted(os.listdir(frame_folder)) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(frame_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(frame_folder, image)))

    video.release()