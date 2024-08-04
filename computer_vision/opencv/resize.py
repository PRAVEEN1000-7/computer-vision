# i ).

import cv2
import imutils

img = cv2.imread('novaicon.png')
resized_img = imutils.resize(img,width=300)

cv2.imshow('original image :',img)
cv2.imshow('resized image :',resized_img)

cv2.waitKey(10000)
cv2.destroyAllWindows()

# ii ).

import cv2

def rescale(frame):
    
    heigth = int(frame.shape[0]*.75)
    width = int(frame.shape[1]*.75)
    size = (width,heigth)
    
    return cv2.resize(frame,size,interpolation=cv2.INTER_AREA)

video = cv2.VideoCapture(0)

while True :
    true,frame = video.read()
    
    cv2.imshow('resized live',rescale(frame))
    
    if cv2.waitKey(1)==27: # 27 --> ascii value of esc key ( click esc key to close )
        exit(0)


