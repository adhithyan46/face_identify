import math

class Tracker :
    def __init__(self):
        self.center_point ={}
        self.count_id = 0
        
    def update(self,object_rectangle):
        object_bbox_id = []
        
        for rect in object_rectangle: 
            x,y,w,h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2
            
            same_object_detected = False
            for id , pre_track in self.center_point.items():
                dist = math.hypot(cx - pre_track[0],cy - pre_track[1])
                
                if dist < 70:
                    self.center_point[id] = (cx,cy)
                    object_bbox_id.append([x,y,w,h,id])
                    same_object_detected = True
                    break
                
            if same_object_detected is False:
                self.center_point[self.count_id] = (cx,cy)
                object_bbox_id.append([x,y,w,h,self.count_id])
                self.count_id += 1
                
        new_center_points = {}
        for bbox_ids in object_bbox_id:
            _,_,_,_,object_id = bbox_ids
            center = self.center_point[object_id]
            new_center_points[object_id] = center
            
        self.center_point = new_center_points.copy()
        return object_bbox_id