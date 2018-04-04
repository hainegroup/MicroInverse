#!/usr/bin/env python
 
# Haversine formula example in Python
# Author: Wayne Dyck
 
#import math
import numpy as np
 
def distance(origin, destination):
    '''
    origin      = (lat1, lon1)
    destination = (lat2, lon2)
    '''
    #
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371 # km
    #
    dlat = np.radians(lat2-lat1) #math.radians(lat2-lat1)
    dlon = np.radians(lon2-lon1) #math.radians(lon2-lon1)
    a = np.sin(dlat/2) * np.sin(dlat/2) + np.cos(np.radians(lat1))* np.cos(np.radians(lat2)) * np.sin(dlon/2) * np.sin(dlon/2)
    #a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1))* math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a)) #2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c
    
    return d
