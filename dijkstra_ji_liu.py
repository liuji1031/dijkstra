import argparse
import numpy as np
import heapq
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

class Map:
    """class representing the map
    """
    def __init__(self,
                 width=1200,
                 height=500,
                 inflate_radius=5
                 ):
        """create a map object to represent the discretized map

        Args:
            width (int, optional): the width of the map. Defaults to 1200.
            height (int, optional): the height of the map. Defaults to 500.
            inflate_radius (int, optional): the radius of the robot for inflat-
            ing the obstacles. Defaults to 5.
        """
        self.width = width
        self.height = height
        self.map = np.zeros((height, width),dtype=np.int8) # 0: obstacle free
                                                           # 1: obstacle
        self.map_inflate = np.zeros_like(self.map)
        self.inflate_radius = inflate_radius

    def add_obstacle(self, corners : np.ndarray):
        """add obstacle defined by the corner points. the corners should define
        a convex region, not non-convex ones. for non-convex obstacles, need to 
        define it in terms of the union of the convex parts. 

        Args:
            corners (_type_): the corners of the obstacles, defined in the
            clockwise direction. each row represents the (x,y) coordinate of a 
            corner

        Returns:
            _type_: _description_
        """
        obs_map = np.zeros((self.height, self.width),dtype=np.int8)
        obs_map_inflate = np.zeros_like(obs_map)

        # first get a meshgrid of map coordinates
        x, y = np.meshgrid(np.arange(0,self.width), np.arange(0,self.height))
        xy_all = np.hstack((x.flatten()[:,np.newaxis],
                            y.flatten()[:,np.newaxis]))

        if corners.shape[1] != 2:
            corners = corners.reshape((-1,2)) # make sure it's a 2D array
        n = corners.shape[0]
        for i in range(corners.shape[0]):
            j = int((i+1)%n) # the adjacent corner index in clockwise direction

            # get x, y
            x1,y1 = corners[i,:]
            x2,y2 = corners[j,:]

            # get normal direction
            normal_dir = np.arctan2(y2-y1, x2-x1) + np.pi/2
            normal = np.array([np.cos(normal_dir),np.sin(normal_dir)])

            # compute the projection of one of the corner point
            p = np.inner((x1,y1),normal)

            # find the meshgrid points whose projections are <= p
            proj_all = np.inner(xy_all, normal).reshape((self.height,
                                                         self.width))

            obs_map += np.where(proj_all<=p,1,0)
            obs_map_inflate += np.where(proj_all<=p+self.inflate_radius,1,0)
        
        # find points that meet all half plane conditions
        obs_map = np.where(obs_map==n,1,0)
        obs_map_inflate = np.where(obs_map_inflate==n,1,0)

        # add to the existing map
        self.map = np.where(obs_map==1,obs_map,self.map)
        self.map_inflate = np.where(obs_map_inflate==1,
                                    obs_map_inflate,self.map_inflate)