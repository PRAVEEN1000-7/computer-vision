import cv2

img = cv2.imread('novaicon.png')

#syntax : var = cv2.GaussianBlur( source , (kernek) , borderType )
blur1 = cv2.GaussianBlur(img,(21,21),0) # normally used 
blur2 = cv2.GaussianBlur(img,(41,41),0)
blur3= cv2.GaussianBlur(img,(41,41),5)

cv2.imshow('original image :',img)
cv2.imshow('blur 1 image :',blur1)
cv2.imshow('blur 2 image :',blur2)
cv2.imshow('blur 3 image :',blur3)

cv2.waitKey(0)
cv2.destroyAllWindows()
