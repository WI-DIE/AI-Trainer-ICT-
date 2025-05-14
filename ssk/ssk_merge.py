import cv2
import mediapipe as mp
import numpy as np
import time
import pygame
from screeninfo import get_monitors  # 화면 해상도 감지를 위한 라이브러리
from utils_ssk import calculate_angle, draw_angle, crop_to_aspect_ratio, resize_to_fixed_height, get_color_for_angle_0, get_color_for_angle_1, get_gradient_color

# pygame 초기화 및 음성 파일 로드
pygame.mixer.init()
sound = pygame.mixer.Sound('perfect2.mp3')

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 화면 해상도를 자동으로 감지
monitor = get_monitors()[0]
screen_width = monitor.width
screen_height = monitor.height

# Initialize counters and flags
left_counter = 0
right_counter = 0
left_stage = None
right_stage = None
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
INPUT_VIDEO_PATH = "ssk(960x1080).mov"
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

    # 첫 번째 줄: 'Get ready for the'

    cv2.putText(cb_frame, 'READY', (270, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 13, cv2.LINE_AA)
    cv2.putText(cb_frame, 'READY', (270, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 8, cv2.LINE_AA)

    '''cv2.putText(cb_frame, 'Get ready for a', (100, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 9, cv2.LINE_AA)
    cv2.putText(cb_frame, 'Get ready for a', (100, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 5, cv2.LINE_AA)

    # 두 번째 줄: 'Standing Side Knee-Up'
    cv2.putText(cb_frame, 'Standing Side Knee-Up', (100, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 13, cv2.LINE_AA)
    cv2.putText(cb_frame, 'Standing Side Knee-Up', (100, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 8, cv2.LINE_AA)'''

    cv2.imshow("Ready", cb_frame)
    cv2.waitKey(1)

with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:
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
        frame2 = resize_to_fixed_height(frame2, 1080)  # 높이를 1080으로 조정

        # 좌우 반전
        frame1 = cv2.flip(frame1, 1)

        # Recolor image to RGB
        image = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(frame1)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        combined_frame = cv2.hconcat([image, frame2])

        try:
            landmarks = results.pose_landmarks.landmark

            # landmarks setup
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                    landmarks[mp_pose.PoseLandmark.NOSE.value].y]
            left_eye_inner = [landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER.value].y]
            left_eye = [landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].y]
            left_eye_outer = [landmarks[mp_pose.PoseLandmark.LEFT_EYE_OUTER.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_EYE_OUTER.value].y]
            right_eye_inner = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].y]
            right_eye = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].y]
            right_eye_outer = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value].y]
            left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
            right_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]
            mouth_left = [landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].x,
                          landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].y]
            mouth_right = [landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].x,
                           landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].y]
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            left_pinky = [landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].y]
            right_pinky = [landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].y]
            left_index = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y]
            right_index = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y]
            left_thumb = [landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].y]
            right_thumb = [landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].y]
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
            left_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
            right_heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]
            left_foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
            right_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]

            # Calculate angles
            left_angle_0 = calculate_angle(left_elbow, left_shoulder, left_hip)
            right_angle_0 = calculate_angle(
                right_elbow, right_shoulder, right_hip)
            left_angle_1 = calculate_angle(left_shoulder, left_hip, left_knee)
            right_angle_1 = calculate_angle(
                right_shoulder, right_hip, right_knee)

            # setup line color
            line_color_left_0 = get_color_for_angle_0(left_angle_0)
            line_color_left_1 = get_color_for_angle_1(left_angle_1)
            line_color_right_0 = get_color_for_angle_0(right_angle_0)
            line_color_right_1 = get_color_for_angle_1(right_angle_1)

            # Visualize angles
            draw_angle(image, combined_frame, left_shoulder,
                       left_hip, left_knee, left_angle_1)
            draw_angle(image, combined_frame, right_shoulder,
                       right_hip, right_knee, right_angle_1)
            draw_angle(image, combined_frame, left_hip,
                       left_shoulder, left_elbow, left_angle_0)
            draw_angle(image, combined_frame, right_hip,
                       right_shoulder, right_elbow, right_angle_0)

            # Standing side knee up logic for left part
            if right_angle_0 < 85 and right_angle_1 < 50:  # 오른쪽 각도로 변경
                left_stage = "up"
            if right_angle_1 >= 160 and left_stage == 'up':  # 오른쪽 각도로 변경
                left_stage = "down"
                left_counter += 1
                sound.play()
                show_excellent = True
                excellent_start_time = time.time()
                print(f'Left Knee Up Count: {left_counter}')  # 여전히 왼쪽 카운터에 반영

            # Standing side knee up logic for right part
            if left_angle_0 < 85 and left_angle_1 < 50:  # 왼쪽 각도로 변경
                right_stage = "up"
            if left_angle_1 >= 160 and right_stage == 'up':  # 왼쪽 각도로 변경
                right_stage = "down"
                right_counter += 1
                sound.play()
                show_excellent = True
                excellent_start_time = time.time()
                # 여전히 오른쪽 카운터에 반영
                print(f'Right Knee Up Count: {right_counter}')

            # 왼쪽 랜드마크 그리기
            left_angle_0_color = get_color_for_angle_0(left_angle_0)
            left_angle_1_color = get_color_for_angle_1(left_angle_1)

            left_elbow_px = tuple(np.multiply(
                left_elbow, [image.shape[1], image.shape[0]]).astype(int))
            left_shoulder_px = tuple(np.multiply(
                left_shoulder, [image.shape[1], image.shape[0]]).astype(int))
            left_hip_px = tuple(np.multiply(
                left_hip, [image.shape[1], image.shape[0]]).astype(int))
            left_knee_px = tuple(np.multiply(
                left_knee, [image.shape[1], image.shape[0]]).astype(int))
            for point_px in [left_elbow_px, left_shoulder_px, left_hip_px, left_knee_px]:
                cv2.circle(combined_frame, point_px, 10,
                           (50, 50, 200), cv2.FILLED)
                cv2.circle(combined_frame, point_px, 15, (0, 0, 255), 2)
                cv2.line(combined_frame, left_shoulder_px,
                         left_elbow_px, left_angle_0_color, 2)
                cv2.line(combined_frame, left_hip_px,
                         left_knee_px, left_angle_1_color, 2)

            # 오른쪽 랜드마크 그리기
            right_angle_0_color = get_color_for_angle_0(right_angle_0)
            right_angle_1_color = get_color_for_angle_1(right_angle_1)

            right_elbow_px = tuple(np.multiply(
                right_elbow, [image.shape[1], image.shape[0]]).astype(int))
            right_shoulder_px = tuple(np.multiply(
                right_shoulder, [image.shape[1], image.shape[0]]).astype(int))
            right_hip_px = tuple(np.multiply(
                right_hip, [image.shape[1], image.shape[0]]).astype(int))
            right_knee_px = tuple(np.multiply(
                right_knee, [image.shape[1], image.shape[0]]).astype(int))
            for point_px in [right_elbow_px, right_shoulder_px, right_hip_px, right_knee_px]:
                cv2.circle(combined_frame, point_px, 10,
                           (50, 50, 200), cv2.FILLED)
                cv2.circle(combined_frame, point_px, 15, (0, 0, 255), 2)
                cv2.line(combined_frame, right_shoulder_px,
                         right_elbow_px, right_angle_0_color, 2)
                cv2.line(combined_frame, right_hip_px,
                         right_knee_px, right_angle_1_color, 2)

            # Calculate gauge for left side (오른쪽 게이지 바 계산으로 변경)
            right_percent = np.interp(
                ((right_angle_0)+(right_angle_1))/2, (65, 160), (100, 0))
            right_bar_width = int(right_percent * 2)
            bar_x_start, bar_y_start = 10, 600
            bar_x_end, bar_y_end = 310, 630

            bar_color_right = get_gradient_color(right_percent)

            max_bar_width = bar_x_end - bar_x_start

            right_bar_width = int((right_percent / 100) * max_bar_width)

            # Draw gauge for left side (오른쪽 게이지 바를 왼쪽에 그리기)
            cv2.rectangle(combined_frame, (bar_x_start, bar_y_start),
                          (bar_x_end, bar_y_end), (255, 255, 255), 2)
            cv2.rectangle(combined_frame, (bar_x_start, bar_y_start), (bar_x_start +
                                                                       right_bar_width, bar_y_end), bar_color_right, cv2.FILLED)
            cv2.putText(combined_frame, f'{int(right_percent)}%', (bar_x_end + 10, bar_y_start + 23),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 6)
            cv2.putText(combined_frame, f'{int(right_percent)}%', (bar_x_end + 10, bar_y_start + 23),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)

            # Calculate gauge for right side (왼쪽 게이지 바 계산으로 변경)
            left_percent = np.interp(
                ((left_angle_0)+(left_angle_1))/2, (65, 160), (100, 0))
            left_bar_width = int(left_percent * 2)
            bar_y_start_right = 650
            bar_y_end_right = 680

            left_bar_width = int((left_percent / 100) * max_bar_width)
            bar_color_left = get_gradient_color(left_percent)

            # Draw gauge for right side (왼쪽 게이지 바를 오른쪽에 그리기)
            cv2.rectangle(combined_frame, (bar_x_start, bar_y_start_right),
                          (bar_x_end, bar_y_end_right), (255, 255, 255), 2)
            cv2.rectangle(combined_frame, (bar_x_start, bar_y_start_right), (bar_x_start +
                                                                             left_bar_width, bar_y_end_right), bar_color_left, cv2.FILLED)
            cv2.putText(combined_frame, f'{int(left_percent)}%', (bar_x_end + 10, bar_y_start_right + 23),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 6)
            cv2.putText(combined_frame, f'{int(left_percent)}%', (bar_x_end + 10,
                                                                  bar_y_start_right + 23), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)

        except:
            pass

        # LEFT COUNT 텍스트에 하이라이트 추가 (오른쪽 카운터로 변경)
        (left_text_w, left_text_h), _ = cv2.getTextSize(
            'RIGHT COUNT', cv2.FONT_HERSHEY_TRIPLEX, 1, 1)
        cv2.rectangle(combined_frame, (7, 20), (10 + left_text_w,
                                                38 + left_text_h), (0, 0, 0), cv2.FILLED)

        cv2.putText(combined_frame, 'RIGHT COUNT', (10, 50),
                    cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(combined_frame, str(right_counter),
                    (100, 115),
                    cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 220), 2, cv2.LINE_AA)

        # RIGHT COUNT 텍스트에 하이라이트 추가 (왼쪽 카운터로 변경)
        (right_text_w, right_text_h), _ = cv2.getTextSize(
            'LEFT COUNT', cv2.FONT_HERSHEY_TRIPLEX, 1, 1)
        cv2.rectangle(combined_frame, (7, 140), (10 + right_text_w,
                                                 158 + right_text_h), (0, 0, 0), cv2.FILLED)

        cv2.putText(combined_frame, 'LEFT COUNT', (10, 170),
                    cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(combined_frame, str(left_counter),
                    (100, 235),
                    cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 220), 2, cv2.LINE_AA)

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

        # Display the image
        cv2.imshow('Combined Video', combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
