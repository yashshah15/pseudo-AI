import cv2
import os

for i in os.listdir("D:/real-and-fake-face-detection/dataset/training/training_real/"):
	img=cv.imread(i)
	cv2.imwrite('dataset/real/'+i,cv2.resize(img,(128,128)))
	print("image saved")