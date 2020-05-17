import numpy as np
from pyproj import Geod

# Distance calculations ---------------------------------------------------------------------------------------------------------
## Vincenty's Distance
def calc_vincenty_dist(lat1, lon1, lat2, lon2): 
    az12, az21, dist = Geod(ellps='WGS84').inv(lon1, lat1, lon2, lat2)
    return dist

## Manhattan Distance, Cartesian Coordinate - Output: LatLon distance
# def calc_manh_distc(lat, lon):
#     return sum(abs(lat_i-lat_j) for lat_i, lat_j in zip(lat, lon))

# Manhattan Distance, Haversine Distance + Inclination 29 degrees to True north per Manhattan streets
# Credits: https://www.movable-type.co.uk/scripts/latlong.html, https://gist.github.com/jkAtGitHub/8ae7da4d5dacb9969bff43500b5efbc0#file-manhattan_dist-py

def calc_haversine_dist(lat1, lon1, lat2, lon2):
    lat1, lat2, lon1, lon2 = np.radians(lat1), np.radians(lat2), np.radians(lon1), np.radians(lon2)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    r = 3963 # Earth's radius in miles
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a)) 
    
    total_dist = r * c
    return total_dist

def calc_manh_disth(lat1, lon1, lat2, lon2):
    start = np.stack([lat1, lon1], axis=1)
    end = np.stack([lat2, lon2], axis=1)
    
    theta1, theta2 = np.radians(-28.904), np.radians(28.904)
    rmat1 = np.array([[np.cos(theta1), np.sin(theta1)], 
                      [-np.sin(theta1), np.cos(theta1)]])
    rmat2 = np.array([[np.cos(theta2), np.sin(theta2)], 
                      [-np.sin(theta2), np.cos(theta2)]])
    
    start_rot = rmat1 @ start.T
    end_rot = rmat1 @ end.T
    
    hinge = np.stack((start_rot[0,:], end_rot[1,:]))
    hinge_coords = rmat2 @ hinge
    
    manh_dist = calc_haversine_dist(start.T[0], start.T[1], hinge_coords[0], hinge_coords[1]) + calc_haversine_dist(hinge_coords[0], hinge_coords[1], end.T[0], end.T[1])
    return manh_dist
