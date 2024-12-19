import face_recognition
import os
import numpy as np

def train_cnn(train_dir = 'C:/Users/adhithyan/Documents/NESA/AI project/AAMS/AAMS-05-12-24/app/facerec/dataset'):
    known_face = []
    known_name = []
    
    for person_name in os.listdir(train_dir):
       person_folder = os.path.join(train_dir,person_name)
       for image_name in os.listdir(person_folder):
           image_path = os.path.join(person_folder, image_name)
           image = face_recognition.load_image_file('image_path')
           
           
           face_encoding = face_recognition.face_encodings(image)
           if len(face_encoding) >  0:
                known_face.append(face_encoding[0])
                known_name.append(person_name)
                
    return known_name, known_face
               