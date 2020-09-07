import numpy as np
import cv2

#cascade used for the face
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml')

camera = cv2.VideoCapture(0)
camera.set(3,1280) # set Width
camera.set(4,720) # set Height

while 1:
    ret, img = camera.read()
    img = cv2.flip(img, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #defining a function to return a frame in PIL(Python Imaging Library) form
    def get_img():
        retval, photo_cap = camera.read()
        photo_cap = cv2.flip(photo_cap, -1)
        return photo_cap
        
 
    faces = face_cascade.detectMultiScale(
        gray,     #source
        scaleFactor=1.3, #used to create the scale pyramid
        minNeighbors=5,     #A higher number gives lower false positives.
        minSize=(30, 30)   #minimum rectangle size ti be considered a face
    )

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2) #(image, start_point, end_point, color, thickness)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.5,
            minNeighbors=10,
            minSize=(5,5)
            )
        i=1
        for (ex, ey, ew, eh) in eyes:#rectangles for eyes\
            
            #this is where we actually click take the picture
            print("Taking picture...")
            camera_capture = get_img()
            file_name = "test_image"+str(i)+".png"
            i= i+1
            cv2.imwrite(file_name, camera_capture)
            #this highlights the image
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (255,0,0), 1)

    cv2.imshow('Output',img)

    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break

camera.release()
cv2.destroyAllWindows()




