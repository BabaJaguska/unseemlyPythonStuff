# import numpy
# import os
from PyQt4.QtGui import * # GUI, for file dialog
import sys
import cv2 # OpenCV
from matplotlib import pyplot as plt

###########################################################
plt.close("all")
# so this doesn't seem to close previous session
# print(os.getcwd())
# os.chdir("C:/Users/Mini/My Documents")

###########################################################
#### Choose Image to Analyze - DIALOG ####

app = QApplication(sys.argv)
w = QWidget()
dialog = QFileDialog()
dialog.exec_()
file = dialog.selectedFiles()


# Load the chosen image
img = cv2.imread(file[0])
print("Chosen image is of shape: ", img.shape)


##########################################################
#### Calculate COLOR CHANNEL HISTOGRAMS  ####

# default OpenCV color scheme is BGR, rather than RGB. 
# So if you are displaying color images in matplotlib but read them in open cv,
# they will have inverted channels!! plt: rgb, cv2: bgr
# plt.figure(4)
# plt.imshow(img)
# see? but you can do this:

imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.figure(5)
# plt.imshow(imgRGB) 

hists = [cv2.calcHist([imgRGB], [x], None, [256], [0, 255]) for x in range(0, 3)]
# all these arguments need to be in square brackets!!
# the args are: obviously image, channels,mask,histSizeranges
# For color image, you can pass [0], [1] or [2] to calculate histogram of blue,
# green or red channel respectively
# for grayscale, there is only [0]

colors = ('r', 'g', 'b') 
# yes you can obviously iterate a tuple, retard

for i,col in enumerate(colors):
    plt.plot(hists[i],color=col)
    plt.xlim([0,255])
    #plt.ylim([0,5000])
    plt.xlabel("dark <----- Intensity -----> light")
    plt.title("Color histogram: " + col)

###############################################################
##### Calculate gray levels histogram #########

grayImage = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2GRAY)
# Plot the grayscale version of the image
plt.figure(2)
plt.imshow(grayImage, cmap='gray')

# Calculate grayscale histogram
grayHist = cv2.calcHist([grayImage], [0], None, [256], [0, 255])

plt.figure(1)
plt.plot(grayHist, color='k')
plt.xlabel("dark <----- Intensity -----> light")
plt.title("Gray levels histogram")


################################################################
########## THRESHOLDING #############

thresh = 100
a, b = cv2.threshold(grayImage, thresh, 255, cv2.THRESH_BINARY)
# prvi broj je threshold, drugi je value kojom menjas odsecene delove
# ali ovo prikazuje cv2 a plt nece :/
# the output is 2 numbers, a-retval and b-image
# retval has to do with otsu thresholding

# plt.figure(3)
windowName = 'Image thresholded at intensity level: ' + str(thresh);
cv2.imshow(windowName, b)
# cv2.resizeWindow(windowName, 800, 600)
# Ovo kropuje sliku!!! necu to. hocu zoom out
# if you are displaying color images in matplotlib but read them in open cv,
# they will have inverted channels!! plt: rgb, cv2: bgr

plt.show()

