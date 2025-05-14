import cv2
import mediapipe as mp
import numpy as np
import time
import pygame
from screeninfo import get_monitors  # 화면 해상도 감지를 위한 라이브러리
from utils_squat import calculate_angle, draw_angle, crop_to_aspect_ratio, resize_to_fixed_height, get_gradient_color, get_color_for_angle

# pygame 초기화 및 음성 파일 로드
pygame.mixer.init()
sound = pygame.mixer.Sound('perfect_sound1_KOR.wav')

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 화면 해상도를 자동으로 감지
monitor = get_monitors()[0]
screen_width = monitor.width
screen_height = monitor.height

# Initialize counters and flags
counter = 0
stage = None
show_excellent = False
excellent_start_time = 0

# Load the excellent PNG image
excellent_img_path = "excellent4.png"
excellent_img = cv2.imread(excellent_img_path, cv2.IMREAD_UNCHANGED)

# Resize the PNG image to desired size (e.g., 200x100)
# 이미지 사이즈
desired_width = 150
desired_height = 100
excellent_img = cv2.resize(
    excellent_img, (desired_width, desired_height), interpolation=cv2.INTER_AREA)

# Setup Mediapipe instance
INPUT_VIDEO_PATH = "squat(8x9).mov"
video = cv2.VideoCapture(INPUT_VIDEO_PATH)
cap = cv2.VideoCapture(0)

# Set the desired width and height for the webcam capture
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# 준비
ret, video_frame = video.read()

if not ret:
    print("동영상을 읽을 수 없습니다.")
    video_frame = np.zeos((screen_height, screen_width //
                          2, 3), np.uint8)  # 오류날시 빈화면이 나오도록
# 디버깅을 위해 임시로 WINDOW_NORMAL 사용. mac용
cv2.namedWindow("ready", cv2.WINDOW_NORMAL)

# Window에선 위에꺼 주석처리 하고 이거 두개 열면 됨
# cv2.namedWindow("ready", cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty("ready", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
start_time = time.time()

while time.time() - start_time < 10:
    ret_cam, cam_frame = cap.read()
    if not ret_cam:
        print("캠 영상을 읽을 수 없습니다.")
        cam_frame = np.zeos((screen_height, screen_width // 2, 3), np.uint8)
        break

    cam_frame = crop_to_aspect_ratio(cam_frame, 960, 1080)
    video_frame = resize_to_fixed_height(video_frame, 1080)
    cam_frame = cv2.flip(cam_frame, 1)

    camframe = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
    camframe.flags.writeable = False

    camframe.flags.writeable = True
    camframe = cv2.cvtColor(camframe, cv2.COLOR_RGB2BGR)

    cb_frame = cv2.hconcat([camframe, video_frame])

    # 흰색 테두리를 먼저 그리기 (두꺼운 글씨로)
    cv2.putText(cb_frame, 'Get ready for a', (100, 150),
                cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 8, cv2.LINE_AA)

    # 그 위에 검은색 글씨를 그리기 (얇은 글씨로)
    cv2.putText(cb_frame, 'Get ready for a', (100, 150),
                cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 0), 4, cv2.LINE_AA)

    # 두 번째 줄: '운동명'
    cv2.putText(cb_frame, 'Squat', (100, 300),
                cv2.FONT_HERSHEY_COMPLEX, 4, (255, 255, 255), 10, cv2.LINE_AA)
    cv2.putText(cb_frame, 'Squat', (100, 300),
                cv2.FONT_HERSHEY_COMPLEX, 4, (0, 0, 0), 6, cv2.LINE_AA)

    cv2.imshow("Ready", cb_frame)
    cv2.waitKey(1)


with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose:

    # 전체화면
    cv2.namedWindow('Combined Video', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(
        'Combined Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while video.isOpened() and cap.isOpened():
        ret1, frame1 = cap.read()
        ret2, frame2 = video.read()

        if not ret1 or not ret2:
            break

        # frame1을 자르기
        frame1 = crop_to_aspect_ratio(frame1, 960, 1080)  # 16:9 비율로 자르기

        # frame2를 고정된 높이로 조정하여 비율 유지
        frame2 = resize_to_fixed_height(frame2, 1080)  # 높이를 720으로 조정

        # 좌우 반전
        frame1 = cv2.flip(frame1, 1)

        # Recolor image to RGB
        image = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        combined_frame = cv2.hconcat([image, frame2])

        try:
            landmarks = results.pose_landmarks.landmark

            # landmarks setup
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            # Calculate angles
            left_angle = calculate_angle(left_hip, left_knee, left_ankle)
            right_angle = calculate_angle(right_hip, right_knee, right_ankle)

            # setup line color
            line_color_left = get_color_for_angle(left_angle)
            line_color_right = get_color_for_angle(right_angle)

            # Visualize angles
            draw_angle(image, combined_frame, left_hip,
                       left_knee, left_ankle, left_angle)
            draw_angle(image, combined_frame, right_hip,
                       right_knee, right_ankle, right_angle)

            # Squat logic
            if left_angle < 85 and right_angle < 85:
                stage = "up"
            if left_angle >= 160 and right_angle >= 160 and stage == 'up':
                stage = "down"
                counter += 1
                sound.play()
                show_excellent = True
                excellent_start_time = time.time()
                print(f'Squat Count: {counter}')

            '''# Draw landmarks for left leg
            for point in [left_hip, left_knee, left_ankle]:
                point_px = tuple(np.multiply(
                    point, [image.shape[1], image.shape[0]]).astype(int))
                cv2.circle(combined_frame, point_px, 10,
                           (50, 50, 200), cv2.FILLED)
                cv2.circle(combined_frame, point_px, 15, (0, 0, 255), 2)

            # Draw landmarks for right leg
            for point in [right_hip, right_knee, right_ankle]:
                point_px = tuple(np.multiply(
                    point, [image.shape[1], image.shape[0]]).astype(int))
                cv2.circle(combined_frame, point_px, 10,
                           (50, 50, 200), cv2.FILLED)
                cv2.circle(combined_frame, point_px, 15, (0, 0, 255), 2)'''

            # Draw landmarks for left leg
            left_hip_px = tuple(np.multiply(
                left_hip, [image.shape[1], image.shape[0]]).astype(int))
            left_knee_px = tuple(np.multiply(
                left_knee, [image.shape[1], image.shape[0]]).astype(int))
            left_ankle_px = tuple(np.multiply(
                left_ankle, [image.shape[1], image.shape[0]]).astype(int))
            for point_px in [left_hip_px, left_knee_px, left_ankle_px]:
                cv2.circle(combined_frame, point_px, 10,
                           (50, 50, 200), cv2.FILLED)
                cv2.circle(combined_frame, point_px, 15, (0, 0, 255), 2)
                cv2.line(combined_frame, left_hip_px,
                         left_knee_px, line_color_left, 2)
                cv2.line(combined_frame, left_knee_px,
                         left_ankle_px, line_color_left, 2)

            # Draw landmarks for right leg
            right_hip_px = tuple(np.multiply(
                right_hip, [image.shape[1], image.shape[0]]).astype(int))
            right_knee_px = tuple(np.multiply(
                right_knee, [image.shape[1], image.shape[0]]).astype(int))
            right_ankle_px = tuple(np.multiply(
                right_ankle, [image.shape[1], image.shape[0]]).astype(int))
            for point_px in [right_hip_px, right_knee_px, right_ankle_px]:
                cv2.circle(combined_frame, point_px, 10,
                           (50, 50, 200), cv2.FILLED)
                cv2.circle(combined_frame, point_px, 15, (0, 0, 255), 2)
                cv2.line(combined_frame, right_hip_px,
                         right_knee_px, line_color_right, 2)
                cv2.line(combined_frame, right_knee_px,
                         right_ankle_px, line_color_right, 2)

            # Calculate gauge for left leg and right leg
            left_percent = np.interp(left_angle, (80, 160), (100, 0))
            right_percent = np.interp(right_angle, (80, 160), (100, 0))
            percent = (left_percent + right_percent) / 2

            bar_x_start, bar_x_end = 10, 310
            bar_y_start, bar_y_end = 620, 650

            # 바의 최대 넓이
            max_bar_width = bar_x_end - bar_x_start

            # 바의 너비 계산
            bar_width = int((percent / 100) * max_bar_width)

            # 그라데이션 bar
            bar_color = get_gradient_color(percent)

            # Draw gauge for left leg and right leg
            cv2.rectangle(combined_frame, (bar_x_start, bar_y_start),
                          (bar_x_end, bar_y_end), (255, 255, 255), 2)
            cv2.rectangle(combined_frame, (bar_x_start, bar_y_start), (bar_x_start +
                          bar_width, bar_y_end), bar_color, cv2.FILLED)
            cv2.putText(combined_frame, f'{int(percent)}%', (bar_x_end + 10, bar_y_start + 23),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 6)
            cv2.putText(combined_frame, f'{int(percent)}%', (bar_x_end + 10, bar_y_start + 23),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)

        except Exception as e:
            print(e)
            pass

        # Render squat counter
        (left_text_w, left_text_h), _ = cv2.getTextSize(
            'COUNT', cv2.FONT_HERSHEY_TRIPLEX, 2, 2)
        cv2.rectangle(combined_frame, (13, 25), (16 + left_text_w,
                                                 38 + left_text_h), (0, 0, 0), cv2.FILLED)
        cv2.putText(combined_frame, 'COUNT', (16, 72),
                    cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(combined_frame, str(counter),
                    (100, 170),
                    cv2.FONT_HERSHEY_TRIPLEX, 3, (0, 0, 220), 3, cv2.LINE_AA)

        # Display the excellent image in the center of the webcam frame if show_excellent is True
        # 이미지 위치
        if show_excellent:
            excellent_h, excellent_w, _ = excellent_img.shape
            frame_h, frame_w, _ = frame1.shape
            x_offset = frame_w // 2 - excellent_w // 2
            y_offset = frame_h // 2 - excellent_h // 2

            # Ensure offsets are within bounds
            x_offset = max(0, x_offset)
            y_offset = max(0, y_offset)

            # Calculate overlay region
            overlay_region = combined_frame[y_offset:y_offset +
                                            excellent_h, x_offset:x_offset+excellent_w]

            # Check if overlay region is valid
            if overlay_region.shape[1] == excellent_w:
                # Overlay the PNG image with transparency
                for c in range(0, 3):
                    overlay_region[:, :, c] = excellent_img[:, :, c] * (excellent_img[:, :, 3] / 255.0) + \
                        overlay_region[:, :, c] * \
                        (1.0 - excellent_img[:, :, 3] / 255.0)

            if time.time() - excellent_start_time > 0.5:
                show_excellent = False
        cv2.imshow('Combined Video', combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            # Show the combined frame in a window

    cap.release()
    cv2.destroyAllWindows()
