import cv2
import os
import shutil

def video_frame_rate_conversion(input_video_path, output_video_path, target_fps=12):
    temp_image_folder = 'temp_images'

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open the video.")
        return

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

    if frames:
        first_frame = cv2.imread(frames[0])
        height, width, layers = first_frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, target_fps, (width, height))

        for frame_path in frames:
            frame = cv2.imread(frame_path)
            out.write(frame)

        out.release()

    # 임시 이미지 폴더 삭제
    shutil.rmtree(temp_image_folder)

if __name__ == '__main__':
    input_video_path = 'input_video.mp4'
    output_video_path = 'output_video.mp4'
    video_frame_rate_conversion(input_video_path, output_video_path)
