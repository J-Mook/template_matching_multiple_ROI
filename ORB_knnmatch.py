from logging import captureWarnings
import cv2
import numpy as np

size_const = 3

capture = cv2.VideoCapture(0) #Webcam Capture
# capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280/size_const)
# capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720/size_const)

img1 = cv2.imread('temp.png') #template
w, h = img1.shape[0:2]
img1 = cv2.resize(img1, dsize=(int(h/size_const), int(w/size_const)), interpolation=cv2.INTER_AREA)


while(cv2.waitKey(1)):
    ret, img2 = capture.read()        
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # ORB로 서술자 추출 ---①
    detector = cv2.ORB_create()
    #keypoint, Descriptor
    kp1, desc1 = detector.detectAndCompute(gray1, None)
    kp2, desc2 = detector.detectAndCompute(gray2, None)
    # BF-Hamming 생성 ---②
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING2)
    # knnMatch, k=2 ---③
    matches = matcher.knnMatch(desc1, desc2, 2)

    # 첫번재 이웃의 거리가 두 번째 이웃 거리의 ratio% 이내인 것만 추출---⑤
    ratio = 0.7
    good_matches = [first for first,second in matches \
                        if first.distance < second.distance * ratio]
    print('matches:%d/%d' %(len(good_matches),len(matches)))

    res = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, \
                            flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    # 결과 출력                    
    cv2.imshow('Matching', res)
	
cap.release()
cv2.destroyAllWindows()
