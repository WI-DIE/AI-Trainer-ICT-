import numpy as np
import cv2


def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
        np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def draw_angle(image, combined_frame, a, b, c, angle):
    b = np.array(b)
    offset = np.array([-60, -20])  # 텍스트를 오른쪽으로 -60픽셀, 아래로 -20픽셀 이동
    text_position = (int(b[0] * image.shape[1] + offset[0]),
                     int(b[1] * image.shape[0] + offset[1]))

    # 흰색 배경 텍스트 그리기
    cv2.putText(combined_frame, str(int(angle)),
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 6, cv2.LINE_AA)

    # 검은색 텍스트 그리기
    cv2.putText(combined_frame, str(int(angle)),
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    # Draw lines between the points
    cv2.line(combined_frame, tuple(np.multiply(a, [image.shape[1], image.shape[0]]).astype(int)), tuple(
        np.multiply(b, [image.shape[1], image.shape[0]]).astype(int)), (255, 204, 0), 2)
    cv2.line(combined_frame, tuple(np.multiply(b, [image.shape[1], image.shape[0]]).astype(int)), tuple(
        np.multiply(c, [image.shape[1], image.shape[0]]).astype(int)), (255, 204, 0), 2)
    
def resize_with_padding(frame, target_width, target_height):
        h, w = frame.shape[:2]
        aspect_ratio = w / h
        if target_width / target_height > aspect_ratio:
            new_height = target_height
            new_width = int(new_height * aspect_ratio)
        else:
            new_width = target_width
            new_height = int(new_width / aspect_ratio)

        resized_frame = cv2.resize(
            frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        delta_w = target_width - new_width
        delta_h = target_height - new_height
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        color = [0, 0, 0]
        new_frame = cv2.copyMakeBorder(
            resized_frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return new_frame


def crop_to_aspect_ratio(frame, target_width, target_height):
    h, w = frame.shape[:2]
    target_aspect_ratio = target_height / target_width
    frame_aspect_ratio = h / w

    if frame_aspect_ratio > target_aspect_ratio:
        new_height = int(w * target_aspect_ratio)
        start_y = (h - new_height) // 2
        cropped_frame = frame[start_y:start_y+new_height, :]
    else:
        new_width = int(h / target_aspect_ratio)
        start_x = (w - new_width) // 2
        cropped_frame = frame[:, start_x:start_x+new_width]

    return cropped_frame


def resize_to_fixed_height(frame, target_height):
    h, w = frame.shape[:2]
    new_height = target_height
    new_width = int((target_height / h) * w)
    resized_frame = cv2.resize(
        frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_frame



def get_gradient_color(percent):
    if percent <= 50:
        # 빨간색에서 노란색으로 그라데이션
        normalized_percent = percent / 50.0
        r = 255
        g = int(255 * normalized_percent)
        b = 0
    else:
        # 노란색에서 초록색으로 그라데이션
        normalized_percent = (percent - 50) / 50.0
        r = int(255 * (1 - normalized_percent))
        g = 255
        b = 0

    return (b, g, r)


def get_color_for_angle(angle):
    # 70도 이하인 경우 초록색
    if angle <= 70:
        return (0, 255, 0)  # 초록색 (BGR)

    # 160도 이상인 경우 빨간색
    elif angle >= 160:
        return (0, 0, 255)  # 빨간색 (BGR)

    # 70도와 115도 사이의 경우: 초록색에서 노란색으로 그라데이션
    elif 70 < angle <= 115:
        normalized_angle = (angle - 70) / (115 - 70)
        r = int(255 * normalized_angle)
        g = 255
        b = 0

    # 115도와 160도 사이의 경우: 노란색에서 빨간색으로 그라데이션
    else:
        normalized_angle = (angle - 115) / (160 - 115)
        r = 255
        g = int(255 * (1 - normalized_angle))
        b = 0

    # 색상 반환
    return (b, g, r)

