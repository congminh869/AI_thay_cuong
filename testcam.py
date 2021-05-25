import numpy as np
import cv2

cap = cv2.VideoCapture('rtsp://admin:123@123a@192.168.6.32:554/cam/realmonitor?channel=1&subtype=0')

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here


    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
