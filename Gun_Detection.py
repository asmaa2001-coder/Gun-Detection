import numpy as np
import cv2
import matplotlib.pyplot as plt
import imutils
import datetime

get_dataset = cv2.CascadeClassifier('D:/Final year/2nd semster/CV/CV_Project_Gun_Detection/cascade.xml')
cap = cv2.VideoCapture(0)

first_frame = None
gun_exist =False

while cap.isOpened():
    rat,frame = cap.read()
    frame =cv2.resize(frame,(500,500))
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gun_detection = get_dataset.detectMultiScale(gray_frame, 1.3, 20, minSize=(100, 100))

    if len(gun_detection) >0 :
        gun_exist=True
        
    for (x, y, w, h) in gun_detection:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255, 0), 2)
        roi_gray = gray_frame[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        
    if first_frame is None:
    	first_frame = gray_frame
    	continue
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S %p"),
				(10, frame.shape[0] - 10),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.35, (0, 0, 255), 1)
    if gun_exist:
            print("Gun Detected")
            plt.imshow(frame)
            plt.show()
            break
    else:
       	cv2.imshow("Security Feed", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
    	break
cap.release()
cv2.destroyAllWindows()
	
	
        
        

    
    
