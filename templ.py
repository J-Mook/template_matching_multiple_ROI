import cv2
import time
import numpy as np

size_const = 1

capture = cv2.VideoCapture(2) #Webcam Capture
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280/size_const)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720/size_const)

open('time.txt', 'w').close()

while(True):
	start_time = time.time()
	ret, frame = capture.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	template = cv2.imread('temp.png', 0)
	dst = cv2.resize(template, dsize=(640, 480), interpolation=cv2.INTER_AREA)
	w, h = template.shape[::-1]

	# roi_gray = gray

	roi_x = [0, 500, 800, 200]
	roi_y = [100, 0, 400, 400]
	roi_w = [400, 500, 300, 300]
	roi_h = [300, 500, 300, 300]

	res=[]
	stuff = 0

	for i in range(len(roi_x)): # ROI영역별 실행

		if roi_h[i] >= h and roi_w[i] >= w : #템플릿보다 작은 ROI영역 무시

			roi_gray = gray[roi_y[i]:roi_y[i]+roi_h[i], roi_x[i]:roi_x[i]+roi_w[i]]

			cv2.rectangle(frame, (roi_x[i],roi_y[i]), ((roi_x[i]+roi_w[i]-1), (roi_y[i]+roi_h[i]-1)), (0, 255, 0))

			res.append(cv2.matchTemplate(roi_gray, template, cv2.TM_CCOEFF_NORMED))

			threshold = 0.65

			loc = np.where(res[i] >= threshold)

			for pt in zip(*loc[::-1]):
				cv2.rectangle(frame, (pt[0] + roi_x[i], pt[1] + roi_y[i]), (pt[0] + w + roi_x[i], pt[1] + h + roi_y[i]), (0, 0, 255), 2)
				stuff += 1
	
	cv2.imshow('Test',frame)
	# cv2.imshow('ROI',roi_gray)
	
	if stuff > 0:
		with open('time.txt', 'a') as f:
			f.write(str(stuff) + '  ' + str(time.time() - start_time) + '\n')

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break	
	

cap.release()
cv2.destroyAllWindows()		
