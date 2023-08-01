# -*- coding: utf-8 -*-
import scipy.io as scio
import numpy as np
import cv2
from skimage import measure,morphology

def cropheatmap(map_,scale):
    bw =morphology.closing(map_ > 20, morphology.square(3))
    label_image =measure.label(bw,4)
    regions=measure.regionprops(label_image)
    bbox=regions[0].bbox
    area=regions[0].area
    if len(regions)>1:
        for region in regions:
            if region.area>area:
                bbox=region.bbox
                area=region.area
    bb=np.array(bbox)
    bb[0:2]=bb[0:2]/scale
    bb[2:4]=bb[2:4]*scale 
    if bb[0]>63:
        bb[0]=63
    if bb[2]>63:
        bb[2]=63
    if bb[1]>47:
       bb[1]=47
    if bb[3]>47:
       bb[3]=47
    return bb.reshape(2,2)


def getbbox(maps):
    left_map=maps[0]
    right_map=maps[1]
    left_map= cv2.GaussianBlur(left_map, (3, 3), 0)
    right_map=cv2.GaussianBlur(right_map,(3,3),0)
    left_bbox=cropheatmap(left_map,1.1)
    right_bbox=cropheatmap(right_map,1.1)
    return left_bbox,right_bbox
    
    

    
    