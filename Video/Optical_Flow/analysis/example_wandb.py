import wandb
import cv2 as cv
import numpy as np
import pandas as pd
from flow_metrics import *

# wandb 초기화
wandb.init(project='optical_flow', entity='hyuna')

# 비디오 캡처 초기화
cap = cv.VideoCapture('motorcycle_30fps.mp4')
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

# Shi-Tomasi 코너 검출 파라미터
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Lucas-Kanade 광학 흐름 파라미터
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# 마스크 및 색상 초기화
mask = np.zeros_like(old_frame)
color = np.random.randint(0, 255, (100, 3))

# wandb 테이블 초기화
table = wandb.Table(columns=['Frame', 'mean_u', 'mean_v', 'variance_u', 'variance_v', 'magnitude_histogram', 'angle_histogram'])


frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    new_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_count += 1

    # 광학 흐름 계산
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)

    # 좋은 점들 선택
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # 광학 흐름 벡터 계산
        u = good_new[:, 0] - good_old[:, 0]
        v = good_new[:, 1] - good_old[:, 1]

        # 지표 계산
        mean_u, mean_v = calculate_average_flow(u, v)
        magnitude = calculate_magnitude(u, v)
        angle = calculate_angle(u, v)
        hist, bin_edges = calculate_histogram(angle)
        var_u, var_v = calculate_variance(u, v)

        # wandb에 로그 기록
        wandb.log({
            'Frame': frame_count,
            'mean_u': mean_u,
            'mean_v': mean_v,
            'variance_u': var_u,
            'variance_v': var_v,
            'magnitude_histogram': wandb.Histogram(magnitude),
            'angle_histogram': wandb.Histogram(angle)
        })

        # wandb 테이블에 결과 추가
        table.add_data(frame_count, mean_u, mean_v, var_u, var_v, hist.tolist(), hist.tolist())

        # 트랙 및 방향 벡터 그리기
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

            # 방향 벡터 그리기
            direction = (int(a) + int(15 * np.cos(angle[i])), int(b) + int(15 * np.sin(angle[i])))
            frame = cv.arrowedLine(frame, (int(a), int(b)), direction, color[i].tolist(), 2)

        img = cv.add(frame, mask)

        # wandb에 영상 프레임 추가
        wandb.log({'Optical Flow Analysis': wandb.Image(img)})

        # 이전 프레임 및 점 업데이트
        old_gray = new_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

cap.release()
cv.destroyAllWindows()

wandb.log({'optical_flow_table': table})