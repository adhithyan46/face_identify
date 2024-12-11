import datetime
import numpy as np
import cv2

area1 = [(470,83), (470, 386), (289, 386), (289, 83)]
area2 = [(274,83),(274, 386) ,(167, 386), (167, 83)]

movement_tracker = {}
people_entering = {}
people_exiting = {}

def track_movement(name, bbox):
    
    
    global movement_tracker,people_entering,people_exiting
    
    top, right, bottom, left = bbox
    centroid = ((left + right) // 2, (top + bottom) // 2)
    
    
    area1_result = cv2.pointPolygonTest(np.array(area1, np.int32),centroid,False) >= 0
    area2_result = cv2.pointPolygonTest(np.array(area2, np.int32),centroid,False) >= 0
    
    if area1_result:
        movement_tracker[name] = movement_tracker.get(name, []) + ['area1']
        print(f'{name} detected at area1')
    elif area2_result:
        movement_tracker[name] = movement_tracker.get(name, []) + ['area2']
        print(f'{name} detected at area2')

    if movement_tracker.get(name) == [['area1'],['area2']]:
        if name not in people_entering:
            people_entering[name] = datetime.datetime.now()
            print(f'{name} entered at {people_entering[name]}')
            movement_tracker[name] = []
            
    if movement_tracker.get(name) == [['area2'],['area2']]:
        if name not in people_exiting:
            people_exiting[name] = datetime.datetime.now()
            print(f'{name} exited at {people_exiting[name]}')
            movement_tracker[name] = []