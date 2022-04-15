import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('/Users/stefana/Documents/workspace/compare_pdf_to_image/images/hacker4e.jpg')
color = ('b','g','r')
fig = plt.figure("Histograms")
ax = fig.add_subplot(1, 2, 1)
for i,col in enumerate(color):
    histr = cv.calcHist([img],[i],None,[256],[0,256])
    plt.title("Input image")
    plt.plot(histr,color = col)
    plt.xlim([0,256])
ax = fig.add_subplot(1, 2, 2)
img = cv.imread('/Users/stefana/Documents/workspace/compare_pdf_to_image/images/rezized.jpg')
for i,col in enumerate(color):
    histr = cv.calcHist([img],[i],None,[256],[0,256])
    plt.title("Camera image")
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()