import motmetrics as mm 
from scipy.spatial import distance as dist
import numpy as np

def get_centroid(rect):
    x1, y1, x2, y2 = rect
    x = int((x1+x2)/2)
    y = int((y1+y2)/2)
    return (x, y)

class mot_metrics(object):
    def __init__(self):
        self.acc = mm.MOTAccumulator(auto_id=True)
                
    def update(self,new_track,gt_rect):
        gt_centr = []
        gt_ids = []
        
        for rect in gt_rect:
            x1, y1, x2, y2, rect_id = rect
            gt_centr.append( get_centroid([x1, y1, x2, y2]))
            gt_ids.append(rect_id)
            
        pred_centr = [c[-1] for c in new_track.values()]
        pred_id = [c for c in new_track]
        distances = dist.cdist(np.array(gt_centr),np.array(pred_centr))

        self.acc.update(gt_ids,pred_id, distances.tolist())
        
    
    def get_metrics(self):
        mh = mm.metrics.create()
        summary = mh.compute(self.acc, metrics=["num_frames","idf1","idp","idr"], name="acc")
        return summary