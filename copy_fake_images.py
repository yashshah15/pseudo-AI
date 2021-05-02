import cv2
import os

"""l=os.listdir("D:/dataset/casia/CASIA2/Au")
for i in l:
    print(i)
    original=cv2.imread("D:/dataset/casia/CASIA2/Au/"+i)
    copy_image=original.copy()
    cv2.imwrite('dataset/real/'+i,cv2.resize(copy_image,(128,128)))
    print(i)
"""
l=os.listdir("D:/dataset/casia/CASIA2/Tp")
for i in l:
    original=cv2.imread("D:/dataset/casia/CASIA2/Tp/"+i)
    copy_image=original.copy()
    cv2.imwrite('dataset/fake/'+i,cv2.resize(copy_image,(128,128)))
    print(i)