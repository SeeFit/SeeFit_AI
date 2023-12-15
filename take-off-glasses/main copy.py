import cv2
import os
import shutil
import argparse
import torch
from ease_use_cpu import generate_results, smart_mkdir


def video_to_frames(input_video_path, temp_image_folder, target_fps):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open the video.")
        return []

    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    skip_frames = original_fps / target_fps

    if not os.path.exists(temp_image_folder):
        os.makedirs(temp_image_folder)

    frame_num = 0
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 원본 FPS와 타겟 FPS의 비율에 따라 프레임을 건너뛴다.
        if frame_num % round(skip_frames) == 0:
            image_path = os.path.join(temp_image_folder, f'frame_{frame_num:04d}.png')
            cv2.imwrite(image_path, frame)
            frames.append(image_path)

        frame_num += 1

    cap.release()
    return frames


def frames_to_video(frames, output_video_path, target_fps):
    if not frames:
        return

    first_frame = cv2.imread(frames[0])
    height, width, layers = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, target_fps, (width, height))

    for frame_path in frames:
        frame = cv2.imread(frame_path)
        out.write(frame)

    out.release()

if __name__ == '__main__':
    input_video_path = 'D:\\SeeFit_AI\\take-off-glasses\\input_video.mp4'
    output_video_path = 'D:\\SeeFit_AI\\take-off-glasses\\output_video.mp4'
    temp_image_folder = 'D:\\SeeFit_AI\\take-off-glasses\\temp_images'

    # Convert video to frames
    # frames = video_to_frames(input_video_path, temp_image_folder, 12)

    # Define the argparse object outside the 'if frames:' block
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=temp_image_folder, help="input dir")
    parser.add_argument("--save_dir", type=str, default="results", help="result dir")
    parser.add_argument("--img_size", type=int, default=512, help="image sizes for the model")
    parser.add_argument("--ckpt_path", type=str, default="D:\\SeeFit_AI\\take-off-glasses\\ckpt\\pretrained.pt", help="checkpoint of the model")
    args = parser.parse_args()

    # Remove glasses from each frame
    # if frames:
    #     smart_mkdir(args.save_dir)
    #     generate_results(args, "cpu")

    # Convert modified frames back to video

    directory_path = 'D:\\SeeFit_AI\\take-off-glasses\\results'
    frames = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith(('.png', '.jpg'))]
    
    frames_no_glasses = [os.path.join(args.save_dir, os.path.basename(frame)) for frame in frames]
    frames_to_video(frames_no_glasses, output_video_path, 12)

    # Clean up temporary folders
    # shutil.rmtree(temp_image_folder)
    # shutil.rmtree(args.save_dir)