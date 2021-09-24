from logging import captureWarnings
import cv2
import numpy as np

size_const = 1

capture = cv2.VideoCapture(0) #Webcam Capture
# capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280/size_const)
# capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720/size_const)


while(cv2.waitKey(1)):
    ret, img2 = capture.read()
        
    img1 = cv2.imread('temp.png') #template
    # img2 = cv2.imread('../img/figures.jpg') 
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # ORB로 서술자 추출 ---①
    detector = cv2.ORB_create()
    kp1, desc1 = detector.detectAndCompute(gray1, None)
    kp2, desc2 = detector.detectAndCompute(gray2, None)
    # BF-Hamming 생성 ---②
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING2)
    # knnMatch, k=2 ---③
    matches = matcher.match(desc1, desc2)

    # 매칭 결과를 거리기준 오름차순으로 정렬 ---③
    matches = sorted(matches, key=lambda x:x.distance)
    # 모든 매칭점 그리기 ---④
    res1 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, \
                        flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    # 매칭점으로 원근 변환 및 영역 표시 ---⑤
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ])
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ])
    # RANSAC으로 변환 행렬 근사 계산 ---⑥
    mtrx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 8.0)
    h,w = img1.shape[:2]
    pts = np.float32([ [[0,0]],[[0,h-1]],[[w-1,h-1]],[[w-1,0]] ])
    dst = cv2.perspectiveTransform(pts,mtrx)
    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    # 정상치 매칭만 그리기 ---⑦
    matchesMask = mask.ravel().tolist()
    res2 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, \
                        matchesMask = matchesMask,
                        flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    # 모든 매칭점과 정상치 비율 ---⑧
    accuracy=float(mask.sum()) / mask.size
    print("accuracy: %d/%d(%.2f%%)"% (mask.sum(), mask.size, accuracy))

    # 결과 출력                    
    # cv2.imshow('Matching-All', res1)
    cv2.imshow('Matching-Inlier ', res2)
	
cap.release()
cv2.destroyAllWindows()