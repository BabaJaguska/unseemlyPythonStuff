import sys
import cv2
from matplotlib import pyplot as plt

from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog
import numpy as np

###### FILE DIALOG ####

def chooseFile():
    app = QApplication(sys.argv)
    w = QWidget()
    dialog = QFileDialog()
    dialog.exec_()
    filePath = dialog.selectedFiles()
    # vidi da passujes argument: message "choose blabla"
    return filePath[0]


def readAndFixColor(filePath):
    im = cv2.imread(filePath)
    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    return im


####### Blend the cropped and warped into target image ######
def alphaBlend(targetImage, sourceImage, coords):
    #coords consists of:
    ystart, yend, xstart, xend = coords
    
    for y in range(ystart, yend):
        for x in range(xstart, xend):
            if sum(sourceImage[y, x]) >5:
                targetImage[y, x] = sourceImage[y, x]
    return targetImage



sourceFile = chooseFile()
targetFile = chooseFile()
imSrc = readAndFixColor(sourceFile)
imTgt = readAndFixColor(targetFile)

# Pazi ovo ce ti biti lista, a ne string. Pa koristi file[0]


##### display input images ###@

plt.figure(1)
plt.imshow(imSrc)
plt.title("Source Image")

plt.figure(2)
plt.imshow(imTgt)
plt.title("Target Image")


### warp ###

# source coords to warp
top_left = [0, 0]
top_right = [500, 0]
bottom_right = [500, 500]
bottom_left = [0, 500]

## crop the image piece you wanna transform
imSrcCrop = imSrc[top_left[1]:bottom_left[1], top_left[0]:top_right[0]] # img[height,width]


# plot the cropped image
plt.figure(3)
plt.imshow(imSrcCrop)
plt.title("Cropped Image")

# destination coords
top_left_dst = [50, 100]
top_right_dst = [400, 150]
bottom_right_dst = [650, 500]
bottom_left_dst = [100, 450]

# BIG RED FLAG!!!!! ORDER THE POINTS COUNTERCLOCKWISE!!!!
src = np.array([
    [0, bottom_left[1] - top_left[1]],
    [top_right[0] - top_left[0], bottom_left[1] - top_left[1]],
    [top_right[0] - top_left[0], 0],
    [0, 0]],
    dtype="float32")

dst = np.array([
    bottom_left_dst,
    bottom_right_dst,
    top_right_dst,
    top_left_dst],
    dtype="float32")


M = cv2.getPerspectiveTransform(src, dst)

warp = cv2.warpPerspective(imSrcCrop, M, (imSrc.shape[1], imSrc.shape[0])) # inverted coords???
# this warps THE WHOLE IMAGE!!! CROP FIRST??


plt.figure(4)
plt.imshow(warp)
plt.title("Warped Image")


xstart = min(bottom_left_dst[0], top_left_dst[0])
xend = max(top_right_dst[0], bottom_right_dst[0])
ystart = min(top_left_dst[1], top_right_dst[1])
yend = max(bottom_right_dst[1], bottom_left_dst[1])



# remove the black background
coords = [ystart, yend, xstart, xend]
blendano = alphaBlend(imTgt, warp, coords)

plt.figure(5)
plt.imshow(blendano)
plt.title("Blended Image")

plt.show()