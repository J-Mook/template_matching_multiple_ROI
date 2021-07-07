import os
import cv2, math
import numpy as np
import pandas as pd
import pyautogui

img_path = os.path.dirname(os.path.realpath(__file__))
tpl = cv2.imread(os.path.join(img_path, 'temp.png'), cv2.IMREAD_GRAYSCALE)
th, tw = tpl.shape[:2]

cv2.namedWindow('result')
while True:
	screen = pyautogui.screenshot(region=(0, 50, 800, 500))
	src = np.array(screen)
	src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

	res = cv2.matchTemplate(src, tpl, cv2.TM_CCOEFF_NORMED)
	_, _, _, maxloc = cv2.minMaxLoc(res)

	dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
	cv2.rectangle(dst, maxloc, (maxloc[0] + tw, maxloc[1] + th), (0, 255, 0), 3)

	# print(pd.DataFrame(np.array(maxloc), columns = ["x", "y"]))
	print(maxloc)
	cv2.imshow('result', dst)

	if cv2.waitKey(1) == 27:
		break

cv2.destroyAllWindows()
