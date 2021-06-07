# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 10:12:13 2021

@author: msi
"""

import numpy as np
import laspy
import pptk
import os
import glob
#path to LiDAR
#path_to_file="./test"

path_to_predictions="./test"
#random sampling between 0-1
sampling_factor = 0.75
#grid sampling
#voxel_size = 0.5
#factor
factor = 65535
POINT_SIZE = 0.001
def read_LAS(path_to_file,factor):
     # reading las file and copy points
    input_las = laspy.file.File(path_to_file, mode="r")
    point_records = input_las.points.copy()
    # getting the colors
    red=input_las.red
    green=input_las.green
    blue=input_las.blue
    # getting scaling and offset parameters
    las_scaleX = input_las.header.scale[0]
    las_offsetX = input_las.header.offset[0]
    las_scaleY = input_las.header.scale[1]
    las_offsetY = input_las.header.offset[1]
    las_scaleZ = input_las.header.scale[2]
    las_offsetZ = input_las.header.offset[2]

    # calculating coordinates
    p_X = np.array((point_records['point']['X'] * las_scaleX) + las_offsetX)
    p_Y = np.array((point_records['point']['Y'] * las_scaleY) + las_offsetY)
    p_Z = np.array((point_records['point']['Z'] * las_scaleZ) + las_offsetZ)
    points=np.vstack((p_X, p_Y, p_Z)).transpose()
    #stacking into (N,3)
    colors = np.vstack((red/factor, green/factor, blue/factor)).transpose()
    points = np.column_stack((points,colors))
    return points

def random_sampling(points,sampling_factor):
    #decimated_points_random = points[::sampling_factor]
    if sampling_factor > 1:
        sampling_factor = 1
    number_of_rows = points.shape[0]
    random_indices = np.random.choice(number_of_rows, size=int(number_of_rows*sampling_factor), replace=False)

    decimated_points_random = points[random_indices, :]
    return decimated_points_random

def grid_sampling(points,voxel_size):
    #nb_vox = np.ceil((np.max(points, axis=0) - np.min(points, axis=0))/voxel_size)
    non_empty_voxel_keys, inverse, nb_pts_per_voxel = np.unique(((points - np.min(points, axis=0)) // voxel_size).astype(int), axis=0, return_inverse=True, return_counts=True)
    idx_pts_vox_sorted=np.argsort(inverse)
    voxel_grid = {}
    grid_point_sample = []
    grid_candidate_center = []
    last_seen=0
    for idx,vox in enumerate(non_empty_voxel_keys):
        voxel_grid[tuple(vox)]= points[idx_pts_vox_sorted[last_seen:last_seen+nb_pts_per_voxel[idx]]]
        grid_candidate_center.append(
                voxel_grid[tuple(vox)][np.linalg.norm(voxel_grid[tuple(vox)] -
                np.mean(voxel_grid[tuple(vox)],axis=0),axis=1).argmin()])
        last_seen+=nb_pts_per_voxel[idx]
    for i in range(len(grid_candidate_center)):
        grid_point_sample.append(grid_candidate_center[i])
    return np.array(grid_point_sample)    
    

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def get_corners(BB):
    point_1= (BB[0]+BB[3]/2 , BB[1]+BB[4]/2 ,BB[2]+BB[5]/2 )
    point_2= (BB[0]-BB[3]/2 , BB[1]-BB[4]/2 ,BB[2]-BB[5]/2 )
    point_3= (BB[0]+BB[3]/2 , BB[1]-BB[4]/2 ,BB[2]-BB[5]/2 )
    point_4= (BB[0]-BB[3]/2 , BB[1]-BB[4]/2 ,BB[2]+BB[5]/2 )
    point_5= (BB[0]-BB[3]/2 , BB[1]+BB[4]/2 ,BB[2]-BB[5]/2 )
    point_6= (BB[0]+BB[3]/2 , BB[1]-BB[4]/2 ,BB[2]+BB[5]/2 )
    point_7= (BB[0]-BB[3]/2 , BB[1]+BB[4]/2 ,BB[2]+BB[5]/2 )
    point_8= (BB[0]+BB[3]/2 , BB[1]+BB[4]/2 ,BB[2]-BB[5]/2 )
    corners=np.vstack((point_1,point_2,point_3,point_4,point_5,point_6,point_7,point_8))
    ones = np.ones(corners.shape[0])
    corners = np.column_stack((corners,ones))
    return(corners)

def draw_box(corners):
    mlab.plot3d([corners[0,0], corners[5,0]], [corners[0,1], corners[5,1]], [corners[0,2], corners[5,2]], color=(0,1,1), tube_radius=None, line_width=1, figure=fig)
    mlab.plot3d([corners[0,0], corners[6,0]], [corners[0,1], corners[6,1]], [corners[0,2], corners[6,2]], color=(0,1,1), tube_radius=None, line_width=1, figure=fig)
    mlab.plot3d([corners[0,0], corners[7,0]], [corners[0,1], corners[7,1]], [corners[0,2], corners[7,2]], color=(0,1,1), tube_radius=None, line_width=1, figure=fig)
    mlab.plot3d([corners[1,0], corners[2,0]], [corners[1,1], corners[2,1]], [corners[1,2], corners[2,2]], color=(0,1,1), tube_radius=None, line_width=1, figure=fig)
    mlab.plot3d([corners[1,0], corners[3,0]], [corners[1,1], corners[3,1]], [corners[1,2], corners[3,2]], color=(0,1,1), tube_radius=None, line_width=1, figure=fig)
    mlab.plot3d([corners[1,0], corners[4,0]], [corners[1,1], corners[4,1]], [corners[1,2], corners[4,2]], color=(0,1,1), tube_radius=None, line_width=1, figure=fig)
    mlab.plot3d([corners[4,0], corners[6,0]], [corners[4,1], corners[6,1]], [corners[4,2], corners[6,2]], color=(0,1,1), tube_radius=None, line_width=1, figure=fig)
    mlab.plot3d([corners[4,0], corners[7,0]], [corners[4,1], corners[7,1]], [corners[4,2], corners[7,2]], color=(0,1,1), tube_radius=None, line_width=1, figure=fig)
    mlab.plot3d([corners[2,0], corners[7,0]], [corners[2,1], corners[7,1]], [corners[2,2], corners[7,2]], color=(0,1,1), tube_radius=None, line_width=1, figure=fig)
    mlab.plot3d([corners[2,0], corners[5,0]], [corners[2,1], corners[5,1]], [corners[2,2], corners[5,2]], color=(0,1,1), tube_radius=None, line_width=1, figure=fig)
    mlab.plot3d([corners[5,0], corners[3,0]], [corners[5,1], corners[3,1]], [corners[5,2], corners[3,2]], color=(0,1,1), tube_radius=None, line_width=1, figure=fig)
    mlab.plot3d([corners[6,0], corners[3,0]], [corners[6,1], corners[3,1]], [corners[6,2], corners[3,2]], color=(0,1,1), tube_radius=None, line_width=1, figure=fig)

    
    
if __name__ == '__main__':



    for f in (glob.glob(os.path.join(path_to_predictions,'*.npy'))):
        #print(f,s)		

        points = np.load(f)
        print(points.shape)

        data1 = points[:,:3]
        label1 = points[:,3]

        rgb_codes = [[200, 90, 0],
            [255, 0, 0],
            [255, 0, 255],
            [0, 220, 0],
            [0, 200, 255]]

        color = np.zeros((label1.shape[0], 3))
        for i in range(label1.shape[0]):
            color[i,:] = [code/255 for code in rgb_codes[int(label1[i])]]

        
        #s=v.get('selected')
        #v.close()
        v = pptk.viewer(data1[:,:3],color)
        v.set(point_size=POINT_SIZE,show_axis=False,show_info=False)
        #view.wait()
        v.wait()

        v.close()
