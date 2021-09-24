import cv2
import time
import numpy as np
import rospy
from std_msgs.msg import Int32MultiArray

# # #FULL ROI
roi_x = 0
roi_y = 0
roi_w = 1280
roi_h = 720
receive_data = []

roi_x = int(roi_x - roi_w/2 + 640)
roi_y = int(roi_y - roi_h/2 + 360)

def roi_callback(data):
	global receive_data
	receive_data = data.data
	receive_data[0] = int(receive_data[0] - receive_data[2]/2 + 640)
	receive_data[1] = int(receive_data[1] - receive_data[3]/2 + 360)
	# print(receive_data)


def main():
	global receive_data

	rospy.init_node('vision', anonymous=True)
	rospy.Subscriber("ROI_data", Int32MultiArray, roi_callback)

	size_const = 4

	capture = cv2.VideoCapture(0) #Webcam Capture
	capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
	capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

	
	open('time.txt', 'w').close()

	while(True):
		roi_x = receive_data[0]
		roi_y = receive_data[1]
		roi_w = receive_data[2]
		roi_h = receive_data[3]
		start_time = time.time()
		template = cv2.imread('temp.png', 0)
		
		ret, frame = capture.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		
		w, h = template.shape[::-1]
		template = cv2.resize(template, dsize=(int(w/size_const), int(h/size_const)), interpolation=cv2.INTER_AREA)

		res=[]
		stuff = 0

		if roi_h >= h/size_const and roi_w >= w/size_const : #템플릿보다 작은 ROI영역 무시
			roi_gray = gray[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
			# roi_gray = gray[int(roi_y-roi_h/2):int(roi_y+roi_h/2), int(roi_x-roi_w/2):int(roi_x+roi_w/2)]

			cv2.rectangle(frame, (roi_x,roi_y), ((roi_x+roi_w-1), (roi_y+roi_h-1)), (0, 255, 0))

			res = cv2.matchTemplate(roi_gray, template, cv2.TM_CCOEFF_NORMED)

			threshold = 0.3
			loc = np.where((res > threshold) & (res == np.max(res)))
			# loc = np.where(res > threshold)

			for pt in zip(*loc[::-1]):
				cv2.rectangle(frame, (pt[0] + roi_x, pt[1] + roi_y), (pt[0] + int(w/size_const) + roi_x, pt[1] + int(h/size_const) + roi_y), (0, 0, 255), 2)
				# cv2.putText(frame, 'accuracy: ', (pt[0] + roi_x, pt[1] + roi_y + 4), cv2.FONT_HERSHEY_PLAIN, 3.0, (0,0,0))
				stuff += 1

		cv2.imshow('Test',frame)
		cv2.imshow('template',template)
		
		if stuff > 0:
			with open('time.txt', 'a') as f:
				f.write(str(stuff) + '  ' + str(time.time() - start_time) + '\n')

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break	

		rospy.spin()

if __name__ == '__main__':
	main()
	cv2.destroyAllWindows()	