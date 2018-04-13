# import numpy
# import os
from PyQt4.QtGui import *

import sys

import cv2

from matplotlib import pyplot as plt

##############################################
plt.close("all")
# so this doesn't seem to close previous session
# print(os.getcwd())
# os.chdir("C:/Users/Mini/My Documents")

#### pick image ###


app = QApplication(sys.argv)
w = QWidget()
dialog = QFileDialog()
dialog.exec_()
file = dialog.selectedFiles()


img = cv2.imread(file[0])
print(img.shape)


hists = [cv2.calcHist([img], [x], None, [256], [0, 255]) for x in range(0, 3)]
# all this shizz has to be in square brackets
# the args are: obviously image, channels,mask,histSizeranges
# For color image, you can pass [0], [1] or [2] to calculate histogram of blue,
# green or red channel respectively.


colors = ('b', 'g', 'r')
# yes you can obviously iterate a tuple, retard
print(type(hists), len(hists), hists[0].shape)

# for i,col in enumerate(colors):
# 	plt.plot(hists[i],color=col)
# 	plt.xlim([0,255])
# 	#plt.ylim([0,5000])
# 	plt.xlabel('levels')

grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
grayHist = cv2.calcHist([grayImage], [0], None, [256], [0, 255])

plt.figure(1)
plt.plot(grayHist, color='k')
plt.xlabel("dark <----- Intensity -----> light")
plt.title("Gray levels histogram")


plt.figure(2)
plt.imshow(grayImage, cmap='gray')


thresh = 100
a, b = cv2.threshold(grayImage, thresh, 255, cv2.THRESH_BINARY)
# prvi broj je threshold, drugi je value kojom menjas odsecene delove
# ali ovo prikazuje cv2 a plt nece :/
# the output is 2 numbers, a-retval and b-image
# retval has to do with otsu thresholding

# plt.figure(3)
windowName = 'Image thresholded at intensity: ' + str(thresh);
cv2.imshow(windowName, b)
# cv2.resizeWindow(windowName, 800, 600)
# Ovo kropuje sliku!!! necu to. hocu zoom out
# if you are displaying color images in matplotlib but read them in open cv,
# they will have inverted channels!! plt: rgb, cv2: bgr
# plt.figure(4)
# plt.imshow(img)
# see? but you can do this:
# imgRBG=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# plt.figure(5)
# plt.imshow(imgRBG) # i evo ti kako treba

plt.show()
