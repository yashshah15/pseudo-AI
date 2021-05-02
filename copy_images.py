import cv2
import os

for i in os.listdir("D:/real_and_fake_face_detection/real_and_fake_face/training_fake/"):
	img=cv2.imread("D:/real_and_fake_face_detection/real_and_fake_face/training_fake/"+i)
	if i.endswith('resaved.jpg'):
		cv2.imwrite('dataset/fake/'+i,cv2.resize(img,(128,128)))
		print("image saved")