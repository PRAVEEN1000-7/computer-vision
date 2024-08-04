import cv2

img = cv2.imread('novaicon.png')

gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cv2.imwrite('grayscale.png',gray_img)

cv2.imshow('original image :',img)
cv2.imshow('gray scale image :',gray_img)

cv2.waitKey(10000)
cv2.destroyAllWindows()
