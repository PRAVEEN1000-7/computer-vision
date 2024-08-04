import cv2

img=cv2.imread('sample.jpg') # ------>here give your file path with extension  (reading)

cv2.imshow('show',img) #------------> (show the image)
cv2.imwrite('nova.jpg',img) #---------> (write the image)
cv2.waitKey(10000) # --------------->10000 milliseconds(10 sec == 10000 ms)
cv2.destroyAllWindows() #---------> (close the window)
