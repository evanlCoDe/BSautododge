
import cv2
class RoiDetector:
    def __init__(self):
        self.prev=None
        self.binary_img = None
        self.diff_img = None

    def detect(self,frame):
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        if self.prev is None:
            self.prev=gray
            return []
        
        diff=cv2.absdiff(self.prev,gray)
        self.diff_img = diff
        _,th=cv2.threshold(diff,25,255,cv2.THRESH_BINARY)
        self.binary_img = th
        cnts,_=cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        self.prev=gray
        out=[]
        for c in cnts:
            x,y,w,h=cv2.boundingRect(c)
            # if w>h :
            #     if w/2>h:
            #         continue
            # else:
            #     if h/2>w:
            #         continue

            if w*h>100:
                out.append((x,y,w,h))
        return out
