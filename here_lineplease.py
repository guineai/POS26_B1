import cv2
import numpy as np
import os
from PIL import Image

def calculate_steering_angle(image_path, wheelbase_cm):
    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)

    # 이미지 로드
    image = Image.open(image_path)
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # 프레임 다운샘플링 (예: 1/2 크기로 축소)
    frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
    height, width = frame.shape[:2]
    h_step = height // 4
    w_step = width // 4

    # ROI 설정 (2열 4행, 3열 4행, 4열 3행 및 4열 4행만 검출)
    roi_mask = np.zeros((height, width), dtype=np.uint8)
    roi_mask[3 * h_step:4 * h_step, 1 * w_step:2 * w_step] = 255  # 2열 4행
    roi_mask[3 * h_step:4 * h_step, 2 * w_step:3 * w_step] = 255  # 3열 4행
    roi_mask[2 * h_step:3 * h_step, 3 * w_step:4 * w_step] = 255  # 4열 3행
    roi_mask[3 * h_step:4 * h_step, 3 * w_step:4 * w_step] = 255  # 4열 4행

    # 중앙 영역을 추가로 마스킹하여 제외
    central_mask = np.ones((height, width), dtype=np.uint8)
    central_mask[height//3:2*height//3, width//3:2*width//3] = 0
    roi_mask = cv2.bitwise_and(roi_mask, central_mask)

    # 그레이스케일로 변환
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 화이트 필터 적용
    _, white_filter = cv2.threshold(gray_frame, 200, 255, cv2.THRESH_BINARY)

    # 가우시안 블러 적용
    blurred_frame = cv2.GaussianBlur(white_filter, (5, 5), 0)

    # ROI 적용
    gray_frame_roi = cv2.bitwise_and(blurred_frame, blurred_frame, mask=roi_mask)

    # 중간 과정 이미지 저장
    cv2.imwrite(os.path.join(output_dir, "gray_frame.jpg"), gray_frame)
    cv2.imwrite(os.path.join(output_dir, "white_filter.jpg"), white_filter)
    cv2.imwrite(os.path.join(output_dir, "blurred_frame.jpg"), blurred_frame)
    cv2.imwrite(os.path.join(output_dir, "gray_frame_roi.jpg"), gray_frame_roi)

    # 코너 검출
    corners, num_corners = detect_corners(gray_frame_roi)

    # 중앙 영역에서 검출된 코너를 제거
    filtered_corners = []
    for x, y in corners:
        if not (width//3 <= x <= 2*width//3 and height//3 <= y <= 2*height//3):
            filtered_corners.append([x, y])
    filtered_corners = np.array(filtered_corners)

    # 가까운 코너들을 평균값으로 압축
    compressed_corners = compress_corners(filtered_corners, threshold=100)

    # 두 점 사이의 중간점 계산
    midpoints = calculate_midpoints(compressed_corners)

    # 코너 개수 출력
    print(f"Number of corners detected: {len(compressed_corners)}")

    # 원본 프레임에 코너 그리기
    for x, y in compressed_corners:
        cv2.circle(frame, (int(x), int(y)), 20, (0, 0, 255), -1)  # 코너를 빨간색 원으로 표시

    # 원본 프레임에 중간점 그리기
    for x, y in midpoints:
        cv2.circle(frame, (int(x), int(y)), 20, (255, 0, 0), -1)  # 중간점을 파란색 원으로 표시

    # 이미지의 중심점
    center_x, center_y = width // 2, height // 2
    cv2.circle(frame, (center_x, center_y), 20, (255, 0, 255), -1)  # 중심점을 보라색 원으로 표시

    if len(midpoints) == 0:
        raise ValueError("No midpoints detected")

    # 파란 점과 보라 점
    blue_point = midpoints[0]  # 첫 번째 중간점
    purple_point = (center_x, center_y)

    # 두 점 사이의 거리 계산
    d = np.linalg.norm(blue_point - purple_point)

    # 반지름 r
    A = abs(blue_point[0] - purple_point[0])
    B = abs(blue_point[1] - purple_point[1])
    r = (A**2 + B**2) / (2 * A)

    # 두 점의 중간점
    mid_point = ((blue_point[0] + purple_point[0]) / 2, (blue_point[1] + purple_point[1]) / 2)

    # 원의 중심까지의 거리 계산
    a = np.sqrt(r**2 - (d / 2)**2)

    # 원의 중심 계산
    delta_x = a * (purple_point[1] - blue_point[1]) / d
    delta_y = a * (purple_point[0] - blue_point[0]) / d

    center1 = (mid_point[0] + delta_x, mid_point[1] - delta_y)
    center2 = (mid_point[0] - delta_x, mid_point[1] + delta_y)

    # 직교하는 선분을 기준으로 방향 결정
    # 파란 점이 보라 점 기준 왼쪽에 있으면, center1 선택, 아니면 center2 선택
    chosen_center = center1 if blue_point[0] < purple_point[0] else center2

    # 원 그리기
    cv2.circle(frame, (int(chosen_center[0]), int(chosen_center[1])), int(r), (0, 255, 0), 10)  # 초록색으로 원 표시

    # 반지름 r과 조향각 theta 출력
    pixels_to_cm = 24.0 / 1277 
    r_cm = r * pixels_to_cm
    L = wheelbase_cm  # 휠베이스(cm)
    theta = np.arctan(L / r_cm)

    print(f"반지름 r: {r_cm:.2f} cm")
    print(f"조향각 theta: {np.degrees(theta):.2f} degrees")

    # 이미지 축소 비율 조정 (예: 배율 25%)
    scale_percent = 25
    resized_width = int(frame.shape[1] * scale_percent / 100)
    resized_height = int(frame.shape[0] * scale_percent / 100)
    dim = (resized_width, resized_height)

    # 이미지를 축소
    resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    # 결과 이미지 저장
    result_image_path = os.path.join(output_dir, "result.jpg")
    cv2.imwrite(result_image_path, resized_frame)

    return np.degrees(theta), result_image_path

def detect_corners(image):
    gray = np.float32(image)
    dst = cv2.cornerHarris(gray, blockSize=18, ksize=19, k=0.2)
    dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst, 0.4 * dst.max(), 255, 0)
    dst = np.uint8(dst)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    return centroids, len(centroids)

def compress_corners(corners, threshold=100):
    if len(corners) == 0:
        return np.array([])
    compressed_corners = []
    used = np.zeros(len(corners), dtype=bool)
    for i, corner in enumerate(corners):
        if used[i]:
            continue
        close_corners = [corner]
        for j, other_corner in enumerate(corners[i+1:], start=i+1):
            if np.linalg.norm(corner - other_corner) < threshold:
                close_corners.append(other_corner)
                used[j] = True
        compressed_corners.append(np.mean(close_corners, axis=0))
    return np.array(compressed_corners)

def calculate_midpoints(corners):
    midpoints = []
    for i in range(len(corners)):
        for j in range(i + 1, len(corners)):
            midpoint = (corners[i] + corners[j]) / 2
            midpoints.append(midpoint)
    return np.array(midpoints)