from tensorflow.keras.models import load_model
import numpy as np 
import cv2
import os
import facenet
import face_recognition

model = load_model('model-20180402-114759.ckpt-275.data-00000-of-00001')

def get_face_embedding(face_image):
    image = cv2.resize(face_image, (160,160))
    image = np.expand_dims(image, axis=0)
    
    image = image / 255.0
    embedding = model.predict(image)
    return embedding

known_embedding = []
known_name = []

for person_folder in os.listdir('C:/Users/adhithyan/Documents/NESA/AI project/AAMS/AAMS-05-12-24/app/facerec/dataset'):
    for filename in os.listdir('C:/Users/adhithyan/Documents/NESA/AI project/AAMS/AAMS-05-12-24/app/facerec/dataset/{person_folder}'):
        if filename.endswith(('.jpg','png','jpeg')):
            image = cv2.imread('C:/Users/adhithyan/Documents/NESA/AI project/AAMS/AAMS-05-12-24/app/facerec/dataset/{person_folder}/{filename}')
            embedding = get_face_embedding(image)
            known_embedding.append(embedding)
            known_name.append(person_folder)
def identify_face(video_capture):
    while True :
        ret, frame = video_capture.read()
        if not ret:
            print('failed to capture')
            break
        
        face_locations = face_recognition.face_locations(frame)
        for (top,right,left,bottom) in face_locations:
            face_image = frame[top:bottom, left:right]
            
            embedding = get_face_embedding(face_image)
            
            distances = [np.linalg.norm(embedding - known_emb) for known_emb in known_embedding]
            min_distance_index = np.argmin(distances)
            if distances[min_distance_index] < 0.6:
               name = known_name[min_distance_index]
               
            else: 
                name = 'Unknown'
                
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        cv2.imshow('video',frame)
        key = cv2.waitKey(1)
        if key == 27 :
            break
        
    video_capture.release()
    cv2.destroyAllWindows()