import cv2
import mediapipe as mp
import numpy as np
from pushup.utils_pushup import calculate_angle, draw_angle, crop_to_aspect_ratio, resize_to_fixed_height, get_gradient_color

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize counters and flags
counter = 0
stage = None

# Setup Mediapipe instance
INPUT_VIDEO_PATH = "pushup(x3).mov"
video = cv2.VideoCapture(INPUT_VIDEO_PATH)
cap = cv2.VideoCapture(1)

# Set the desired width and height for the webcam capture
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 실제로 설정된 해상도를 확인합니다.
actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Webcam resolution set to: {actual_width}x{actual_height}")

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while video.isOpened() and cap.isOpened():
        ret1, frame1 = cap.read()
        ret2, frame2 = video.read()

        if not ret1 or not ret2:
            break

        # 배율 조정 fx, fy값으로 배율 조정가능!!
        # frame1 = cv2.resize(frame1, (0,0) ,fx = 1.0, fy = 1.0, interpolation=cv2.INTER_AREA)
        # frame2 = cv2.resize(frame2, (0,0) ,fx = 1.0, fy = 1.0, interpolation=cv2.INTER_AREA)

        # frame1 = cv2.resize(frame1, (640, 480))
        # frame2 = cv2.resize(frame2, (640, 480))

        # Resize frames while maintaining aspect ratio
        '''target_height = 1080
        target_width = 960

        frame1 = resize_with_padding(frame1, target_width, target_height)
        frame2 = resize_with_padding(frame2, target_width, target_height)'''
        
        # frame1을 자르기
        frame1 = crop_to_aspect_ratio(frame1, 640, 720)  # 16:9 비율로 자르기

        # frame2를 고정된 높이로 조정하여 비율 유지
        frame2 = resize_to_fixed_height(frame2, 720)  # 높이를 720으로 조정

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
            left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_angle = calculate_angle(
                right_shoulder, right_elbow, right_wrist)
            left_core = calculate_angle(left_shoulder, left_hip, left_knee)
            right_core = calculate_angle(right_shoulder, right_hip, right_knee)
            
            
            def get_color_for_angle(angle):
                if angle < 110:
                    return (0, 255, 0)   # 초록색
                elif 110 <= angle < 160:
                    return (255, 0, 0)   # 파란색
                else:
                    return (0, 0, 255)   # 빨간색
                
            # setup line color
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
                print(f'Push Up Count: {counter}')

            # Draw landmarks for left arm
            left_shoulder_px = tuple(np.multiply(left_shoulder, [image.shape[1], image.shape[0]]).astype(int))
            left_elbow_px = tuple(np.multiply(left_elbow, [image.shape[1], image.shape[0]]).astype(int))
            left_wrist_px = tuple(np.multiply(left_wrist, [image.shape[1], image.shape[0]]).astype(int))
            for point_px in [left_shoulder_px, left_elbow_px, left_wrist_px]:
                cv2.circle(combined_frame, point_px, 10,
                           (50, 50, 200), cv2.FILLED)
                cv2.circle(combined_frame, point_px, 15, (0, 0, 255), 2)
                cv2.line(combined_frame, left_shoulder_px, left_elbow_px, line_color_left, 2)
                cv2.line(combined_frame, left_elbow_px, left_wrist_px, line_color_left, 2)

            # Draw landmarks for right arm
            right_shoulder_px = tuple(np.multiply(right_shoulder, [image.shape[1], image.shape[0]]).astype(int))
            right_elbow_px = tuple(np.multiply(right_elbow, [image.shape[1], image.shape[0]]).astype(int))
            right_wrist_px = tuple(np.multiply(right_wrist, [image.shape[1], image.shape[0]]).astype(int))
            for point_px in [right_shoulder_px, right_elbow_px, right_wrist_px]:
                cv2.circle(combined_frame, point_px, 10,
                           (50, 50, 200), cv2.FILLED)
                cv2.circle(combined_frame, point_px, 15, (0, 0, 255), 2)
                cv2.line(combined_frame, right_shoulder_px, right_elbow_px, line_color_right, 2)
                cv2.line(combined_frame, right_elbow_px, right_wrist_px, line_color_right, 2)

            '''# Calculate gauge for left arm and right arm
            left_percent = np.interp(left_angle, (80, 160), (100, 0))
            right_percent = np.interp(right_angle, (80, 160), (100, 0))
            percent = (left_percent + right_percent)/2
            bar_width = int(percent * 2)

            bar_x_start, bar_x_end = 10, 310
            bar_y_start, bar_y_end = 620, 650
            #bar_color = (0, 255, 0) if percent >= 50 else (0, 0, 255)
            bar_color = get_gradient_color(percent)

            # Draw gauge for left arm and right arm
            cv2.rectangle(combined_frame, (bar_x_start, bar_y_start),
                          (bar_x_end, bar_y_end), (255, 255, 255), 2)
            cv2.rectangle(combined_frame, (bar_x_start, bar_y_start), (bar_x_start +
                          bar_width, bar_y_end), bar_color, cv2.FILLED)
            cv2.putText(combined_frame, f'{int(percent)}%', (bar_x_end + 10,
                        bar_y_start + 23), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)'''
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
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    
        except:
            pass

        # Render push up counter
        '''cv2.putText(combined_frame, 'COUNT', (10, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(combined_frame, str(counter),
                    (10, 110),
                    cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)'''
        (left_text_w, left_text_h), _ = cv2.getTextSize('COUNT', cv2.FONT_HERSHEY_COMPLEX, 1, 2)
        cv2.rectangle(combined_frame, (10, 25), (13 + left_text_w,
                                                 40 + left_text_h), (0, 0, 0), cv2.FILLED)

        cv2.putText(combined_frame, 'COUNT', (13, 55),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(combined_frame, str(counter),
                    (47, 115),
                    cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 220), 3, cv2.LINE_AA)

        # Display the image
        cv2.imshow('Combined Video', combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
