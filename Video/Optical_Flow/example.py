import cv2 as cv
import numpy as np

def shi_tomasi_corner_detection(image, max_corners, quality_level, min_distance):
    corners = cv.goodFeaturesToTrack(image, maxCorners=max_corners, qualityLevel=quality_level, minDistance=min_distance)
    return np.int0(corners)

def lucas_kanade_optical_flow(old_gray, new_gray, p0, window_size=5):
    half_window = window_size // 2 
    Ix = cv.Sobel(old_gray, cv.CV_64F, 1, 0, ksize=5)
    Iy = cv.Sobel(old_gray, cv.CV_64F, 0, 1, ksize=5)
    It = new_gray - old_gray

    u = np.zeros_like(old_gray)
    v = np.zeros_like(old_gray)

    for i in range(half_window, old_gray.shape[0] - half_window):
        for j in range(half_window, old_gray.shape[1] - half_window):
            Ix_window = Ix[i-half_window:i+half_window+1, j-half_window:j+half_window+1].flatten()
            Iy_window = Iy[i-half_window:i+half_window+1, j-half_window:j+half_window+1].flatten()
            It_window = It[i-half_window:i+half_window+1, j-half_window:j+half_window+1].flatten()

            A = np.vstack((Ix_window, Iy_window)).T
            b = -It_window

            nu = np.linalg.pinv(A.T @ A) @ A.T @ b
            u[i, j] = nu[0]
            v[i, j] = nu[1]

    return u, v

cap = cv.VideoCapture('football_3sc.mp4')
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = shi_tomasi_corner_detection(old_gray, 100, 0.3, 7)

mask = np.zeros_like(old_frame)
color = np.random.randint(0, 255, (100, 3))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    new_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    u, v = lucas_kanade_optical_flow(old_gray, new_gray, p0)

    for i, (x, y) in enumerate(p0[:, 0, :]):
        x = int(x)
        y = int(y)
        new_x = int(x + u[y, x])
        new_y = int(y + v[y, x])
        mask = cv.line(mask, (x, y), (new_x, new_y), color[i].tolist(), 2)
        frame = cv.circle(frame, (new_x, new_y), 5, color[i].tolist(), -1)
    
    img = cv.add(frame, mask)
    cv.imshow('frame', img)

    k = cv.waitKey(30) & 0xff
    if k == 27:  # ESC 키를 누르면 종료
        break

    old_gray = new_gray.copy()
    p0 = shi_tomasi_corner_detection(new_gray, 100, 0.3, 7)  # 각 프레임마다 새로운 특징점을 찾음

cap.release()
cv.destroyAllWindows()
