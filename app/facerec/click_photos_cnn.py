import cv2
import os
import face_recognition

def photo_click(dirName, dirID, cam, cnn_model):
    img_counter = 0  
    
    DIR = f"app/facerec/dataset/{dirName}_{dirID}"
    
    try:
        os.mkdir(DIR)
        print("Directory",dirName,"created ")
        
    except FileExistsError:
        print("Directory",dirName,"already exists")
        img_counter = len(os.listdir(DIR))
        
    
    while True:
        ret, frame = cam.read()
        
        if not ret:
            break
        
        cv2.imshow('video',frame)
        key = cv2.waitKey(1)
        
        if key % 256 == 27:
            print('closing ....')
            break
        elif key % 256 ==32:
            img_name = f"app/facerec/dataset/{dirName}_{dirID}/opencv_frame_{img_counter}.png"
            cv2.imwrite(img_name,frame)
            print('{} written !'.format(img_name))
            
            face_locations = face_recognition.face_locations(frame)
            
            for face_loc in face_locations:
                top,right,bottom, left = face_loc
                face_image = frame[top:bottom,left:right]
                