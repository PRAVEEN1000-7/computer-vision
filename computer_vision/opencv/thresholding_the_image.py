import cv2 as cv
img = cv.imread('novaicon.png')

grayImg = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

thresholding = cv.threshold(grayImg,17,255,cv.THRESH_BINARY)[1]

cv.imshow('Original image :',img)
cv.imshow('thresholded image :',thresholding)

cv.waitKey(0)
cv.destroyAllWindows()
