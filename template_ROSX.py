import cv2
import time
import numpy as np


def main():

	capture = cv2.VideoCapture(0) #Webcam Capture
	capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
	capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

	open('time.txt', 'w').close()

	size_const = 4

	while(True):
		roi_x = 0
		roi_y = 0
		roi_w = 1280
		roi_h = 720

		template = cv2.imread('temp.png', 0)
		ret, frame = capture.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		w, h = template.shape[::-1]
		
		template = cv2.resize(template, dsize=(int(w/size_const), int(h/size_const)), interpolation=cv2.INTER_AREA)
		res=[]
		stuff = 0
		
		# set ROI
		roi_gray = gray[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

		cv2.rectangle(frame, (roi_x,roi_y), ((roi_x+roi_w-1), (roi_y+roi_h-1)), (0, 255, 0))

		res = cv2.matchTemplate(roi_gray, template, cv2.TM_CCOEFF_NORMED)

		threshold = 0.3
		loc = np.where((res > threshold) & (res == np.max(res)))
		# loc = np.where(res > threshold)

		for pt in zip(*loc[::-1]):
			inc_w = 120
			inc_h = 120
			roi_x = int((pt[0] + roi_x) - inc_w)
			roi_y = int((pt[1] + roi_y) - inc_h)
			roi_w = int(w/size_const + (inc_w*2))
			roi_h = int(h/size_const + (inc_h*2))
			print((w/size_const * h/size_const)/(roi_w * roi_h))
			cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 4)


		# template matching
		start_time = time.time()

		roi_gray = gray[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
		res = cv2.matchTemplate(roi_gray, template, cv2.TM_CCOEFF_NORMED)

		threshold = 0.3
		loc = np.where((res > threshold) & (res == np.max(res)))
		# loc = np.where(res > threshold)

		for pt in zip(*loc[::-1]):
			cv2.rectangle(frame, (pt[0] + roi_x, pt[1] + roi_y), (pt[0] + int(w/size_const) + roi_x, pt[1] + int(h/size_const) + roi_y), (0, 0, 255), 2)
			stuff += 1

		cv2.imshow('Test',frame)
		cv2.imshow('template',template)
		
		if stuff > 0:
			with open('time.txt', 'a') as f:
				f.write(str(time.time() - start_time) + '\n')

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break	

if __name__ == '__main__':
	main()
	cv2.destroyAllWindows()	