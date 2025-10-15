import cv2 as cv
import numpy as np

# 비디오 캡처 초기화
cap = cv.VideoCapture('kickboard_25fps.mp4')
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

while True:
    ret, frame = cap.read()
    if not ret:
        break
    new_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # 광학 흐름 계산
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)

    # 좋은 점들 선택
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # 트랙 그리기
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        img = cv.add(frame, mask)

        cv.imshow('frame', img)
    
    k = cv.waitKey(30) & 0xff
    if k == 27:  # ESC 키를 누르면 종료
        break

    # 이전 프레임 및 점 업데이트
    old_gray = new_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cap.release()
cv.destroyAllWindows()
