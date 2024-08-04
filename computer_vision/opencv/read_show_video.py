import cv2

video = cv2.VideoCapture(0) #---> (0--> using primary camera | 1--> using secondary camera ) or you may give file path for specific video on your computer.

while True :
    istrue,frame = video.read() #---> gettting the video frame by frame.
    
    cv2.imshow('using front camera :',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('c'): #---> if 'd' is clicked , then close the window
        break

video.release() #--> close the video
cv2.destroyAllWindows()
