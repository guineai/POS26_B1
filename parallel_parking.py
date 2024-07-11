import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def calculate_steering_angle(image_path, wheelbase_cm):
    # 해리스 코너 검출 함수
    def detect_corners(image):
        gray = np.float32(image)
        dst = cv2.cornerHarris(gray, blockSize=18, ksize=19, k=0.2)  # 파라미터 조정
        dst = cv2.dilate(dst, None)
        ret, dst = cv2.threshold(dst, 0.4 * dst.max(), 255, 0)  # 임계값 증가
        dst = np.uint8(dst)
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
        return centroids, len(centroids)  # 코너 개수 반환

    # 가까운 코너들을 평균값으로 압축하는 함수
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

    # 두 점 사이의 중간점을 계산하는 함수
    def calculate_midpoints(corners):
        midpoints = []
        for i in range(len(corners)):
            for j in range(i + 1, len(corners)):
                midpoint = (corners[i] + corners[j]) / 2
                midpoints.append(midpoint)
        return np.array(midpoints)

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

    midpoints = []
    if len(compressed_corners) == 2:
        # 두 점 사이의 중간점 계산
        midpoints = calculate_midpoints(compressed_corners)
    
    # 코너 개수 출력
    print(f"Number of corners detected: {len(compressed_corners)}")

    # 원본 프레임에 코너 그리기
    for x, y in compressed_corners:
        cv2.circle(frame, (int(x), int(y)), 10, (0, 0, 255), -1)  # 코너를 빨간색 원으로 표시

    # 원본 프레임에 중간점 그리기
    for x, y in midpoints:
        cv2.circle(frame, (int(x), int(y)), 10, (255, 0, 0), -1)  # 중간점을 파란색 원으로 표시

    # 이미지의 중심점
    center_x, center_y = width // 2, height // 2
    cv2.circle(frame, (center_x, center_y), 10, (255, 0, 255), -1)  # 중심점을 보라색 원으로 표시

    # x좌표 차이 'A'와 y좌표 차이 'B' 계산
    A = midpoints[0][0] - center_x
    B = midpoints[0][1] - center_y

    # 반지름 r 계산
    r = (A**2 + B**2) / (2*A)

    # 픽셀을 cm로 변환 (가정: 이미지 너비가 24cm일 때)
    pixels_to_cm = 24.0 / width
    r_cm = r * pixels_to_cm

    # 휠베이스 L과 조향각 theta 계산
    L = wheelbase_cm  # cm
    theta = np.arctan(L / r_cm)

    # 반지름 r과 조향각 theta 출력
    print(f"반지름 r: {r_cm:.2f} cm")
    print(f"조향각 theta: {np.degrees(theta):.2f} degrees")

    # 결과 이미지를 저장
    result_image_path = "result_image.jpg"
    cv2.imwrite(result_image_path, frame)

    return np.degrees(theta), result_image_path


