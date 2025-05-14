import cv2
import mediapipe as mp
import numpy as np
import time
import pygame
from screeninfo import get_monitors  # 추가: 화면 정보를 가져오기 위한 모듈
from utils_pushup import calculate_angle, draw_angle, crop_to_aspect_ratio, resize_to_fixed_height, get_gradient_color, get_color_for_angle


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
INPUT_VIDEO_PATH = "pushup(x3).mov"
video = cv2.VideoCapture(INPUT_VIDEO_PATH)
cap = cv2.VideoCapture(0)

# Set the desired width and height for the webcam capture
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# 준비
ret, video_frame = video.read()

if not ret:
    print("동영상을 읽을 수 없습니다.")
    video_frame = np.zeros(
        (screen_height, screen_width // 2, 3), np.uint8)  # 오타 수정

# 디버깅을 위해 임시로 WINDOW_NORMAL 사용. mac용
cv2.namedWindow("ready", cv2.WINDOW_NORMAL)

# Window에선 위에꺼 주석처리 하고 이거 두개 열면 됨
# cv2.namedWindow("ready", cv2.WND_PROP_FULLSCREN)E
# cv2.setWindowProperty("ready", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
start_time = time.time()

while time.time() - start_time < 10:
    ret_cam, cam_frame = cap.read()
    if not ret_cam:
        print("캠 영상을 읽을 수 없습니다.")
        cam_frame = np.zeros(
            (screen_height, screen_width // 2, 3), np.uint8)  # 오타 수정
        break

    # 프레임 처리
    cam_frame = crop_to_aspect_ratio(cam_frame, 960, 1080)
    video_frame = resize_to_fixed_height(video_frame, 1080)
    cam_frame = cv2.flip(cam_frame, 1)

    camframe = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
    camframe.flags.writeable = False
    camframe.flags.writeable = True
    camframe = cv2.cvtColor(camframe, cv2.COLOR_RGB2BGR)

    cb_frame = cv2.hconcat([camframe, video_frame])

    # Ready 텍스트
    cv2.putText(cb_frame, 'READY', (270, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 13, cv2.LINE_AA)
    cv2.putText(cb_frame, 'READY', (270, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 8, cv2.LINE_AA)
    '''# 흰색 테두리를 먼저 그리기 (두꺼운 글씨로)
    cv2.putText(cb_frame, 'Get ready for a', (100, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 9, cv2.LINE_AA)

    # 그 위에 검은색 글씨를 그리기 (얇은 글씨로)
    cv2.putText(cb_frame, 'Get ready for a', (100, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 5, cv2.LINE_AA)

    # 두 번째 줄: '운동명'
    cv2.putText(cb_frame, 'Push-Up', (100, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 13, cv2.LINE_AA)
    cv2.putText(cb_frame, 'Push-Up', (100, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 8, cv2.LINE_AA)'''

    cv2.imshow("ready", cb_frame)
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
        frame2 = resize_to_fixed_height(frame2, 1080)  # 높이를 960으로 조정

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
            left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_angle = calculate_angle(
                right_shoulder, right_elbow, right_wrist)
            left_core = calculate_angle(left_shoulder, left_hip, left_knee)
            right_core = calculate_angle(right_shoulder, right_hip, right_knee)

            # 라인 색
            line_color_left = get_color_for_angle(left_angle)
            line_color_right = get_color_for_angle(right_angle)

            # Visualize angles
            draw_angle(image, combined_frame, left_shoulder,
                       left_elbow, left_wrist, left_angle)
            draw_angle(image, combined_frame, right_shoulder,
                       right_elbow, right_wrist, right_angle)

            # Push up logic
            if left_angle < 70 and right_angle < 70 and left_core >= 160 and right_core >= 160:
                stage = "up"
            if left_angle >= 160 and right_angle >= 160 and stage == 'up' and left_core >= 160 and right_core >= 160:
                stage = "down"
                counter += 1
                sound.play()
                show_excellent = True
                excellent_start_time = time.time()
                print(f'Push Up Count: {counter}')

            '''# Draw landmarks for left arm
            for point in [left_shoulder, left_elbow, left_wrist]:
                point_px = tuple(np.multiply(
                    point, [image.shape[1], image.shape[0]]).astype(int))
                cv2.circle(combined_frame, point_px, 10,
                           (50, 50, 200), cv2.FILLED)
                cv2.circle(combined_frame, point_px, 15, (0, 0, 255), 2)

            # Draw landmarks for right arm
            for point in [right_shoulder, right_elbow, right_wrist]:
                point_px = tuple(np.multiply(
                    point, [image.shape[1], image.shape[0]]).astype(int))
                cv2.circle(combined_frame, point_px, 10,
                           (50, 50, 200), cv2.FILLED)
                cv2.circle(combined_frame, point_px, 15, (0, 0, 255), 2)'''
            # Draw landmarks for left arm
            left_shoulder_px = tuple(np.multiply(
                left_shoulder, [image.shape[1], image.shape[0]]).astype(int))
            left_elbow_px = tuple(np.multiply(
                left_elbow, [image.shape[1], image.shape[0]]).astype(int))
            left_wrist_px = tuple(np.multiply(
                left_wrist, [image.shape[1], image.shape[0]]).astype(int))
            for point_px in [left_shoulder_px, left_elbow_px, left_wrist_px]:
                cv2.circle(combined_frame, point_px, 10,
                           (50, 50, 200), cv2.FILLED)
                cv2.circle(combined_frame, point_px, 15, (0, 0, 255), 2)
                cv2.line(combined_frame, left_shoulder_px,
                         left_elbow_px, line_color_left, 2)
                cv2.line(combined_frame, left_elbow_px,
                         left_wrist_px, line_color_left, 2)

            # Draw landmarks for right arm
            right_shoulder_px = tuple(np.multiply(
                right_shoulder, [image.shape[1], image.shape[0]]).astype(int))
            right_elbow_px = tuple(np.multiply(
                right_elbow, [image.shape[1], image.shape[0]]).astype(int))
            right_wrist_px = tuple(np.multiply(
                right_wrist, [image.shape[1], image.shape[0]]).astype(int))
            for point_px in [right_shoulder_px, right_elbow_px, right_wrist_px]:
                cv2.circle(combined_frame, point_px, 10,
                           (50, 50, 200), cv2.FILLED)
                cv2.circle(combined_frame, point_px, 15, (0, 0, 255), 2)
                cv2.line(combined_frame, right_shoulder_px,
                         right_elbow_px, line_color_right, 2)
                cv2.line(combined_frame, right_elbow_px,
                         right_wrist_px, line_color_right, 2)

            # Calculate gauge for left arm and right arm
            left_percent = np.interp(left_angle, (80, 160), (100, 0))
            right_percent = np.interp(right_angle, (80, 160), (100, 0))
            percent = (left_percent + right_percent) / 2

            # 바의 시작과 끝 좌표
            bar_x_start, bar_x_end = 10, 310
            bar_y_start, bar_y_end = 620, 650

            # 바의 최대 너비 계산
            max_bar_width = bar_x_end - bar_x_start

            # 바의 너비 계산
            bar_width = int((percent / 100) * max_bar_width)

            # 그라데이션 색상 계산
            bar_color = get_gradient_color(percent)

            # Draw gauge for left arm and right arm
            cv2.rectangle(combined_frame, (bar_x_start, bar_y_start),
                          (bar_x_end, bar_y_end), (255, 255, 255), 2)
            cv2.rectangle(combined_frame, (bar_x_start, bar_y_start),
                          (bar_x_start + bar_width, bar_y_end), bar_color, cv2.FILLED)
            cv2.putText(combined_frame, f'{int(percent)}%', (bar_x_end + 10, bar_y_start + 23),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 6)
            cv2.putText(combined_frame, f'{int(percent)}%', (bar_x_end + 10, bar_y_start + 23),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)

        except:
            pass

        # Render push up counter
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

        # Display the image
        cv2.imshow('Combined Video', combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
