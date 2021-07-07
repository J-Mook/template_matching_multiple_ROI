from logging import captureWarnings
import cv2
import numpy as np

size_const = 1

capture = cv2.VideoCapture(0) #Webcam Capture
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280/size_const)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720/size_const)


while(cv2.waitKey(1)):
    ret, img2 = capture.read()
        
    img1 = cv2.imread('temp.png') #template
    # img2 = cv2.imread('../img/figures.jpg') 
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # SIFT 생성
    detector = cv2.xfeatures2d.SIFT_create()
    # 키 포인트와 서술자 추출
    kp1, desc1 = detector.detectAndCompute(gray1, None)
    kp2, desc2 = detector.detectAndCompute(gray2, None)

    # 인덱스 파라미터와 검색 파라미터 설정 ---①
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    # Flann 매처 생성 ---③
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    # 매칭 계산 ---④
    matches = matcher.match(desc1, desc2)
    # 매칭 그리기
    res = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, \
                    flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    # cv2.imshow('Flann + SIFT', res)


    # 매칭 결과를 거리기준 오름차순으로 정렬 ---③
    matches = sorted(matches, key=lambda x:x.distance)
    # 최소 거리 값과 최대 거리 값 확보 ---④
    min_dist, max_dist = matches[0].distance, matches[-1].distance
    # 최소 거리의 15% 지점을 임계점으로 설정 ---⑤
    ratio = 0.1
    good_thresh = (max_dist - min_dist) * ratio + min_dist
    # 임계점 보다 작은 매칭점만 좋은 매칭점으로 분류 ---⑥
    good_matches = [m for m in matches if m.distance < good_thresh]
    print('matches:%d/%d, min:%.2f, max:%.2f, thresh:%.2f' \
            %(len(good_matches),len(matches), min_dist, max_dist, good_thresh))
    # 좋은 매칭점만 그리기 ---⑦
    res = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, \
                    flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    # 결과 출력
    cv2.imshow('Good Match', res)
	
cap.release()
cv2.destroyAllWindows()
