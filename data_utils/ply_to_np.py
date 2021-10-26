# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 10:12:13 2021

@author: msi
"""
import numpy as np
import os
import glob
from plyfile import PlyData

path_to_file="./16_Zurich_16_34322_-22951/"


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc      




if __name__ == '__main__':


    data=np.empty((0,7))

    i=0

    total_seen_class = [0 for _ in range(5)]
    for file in glob.glob(os.path.join(path_to_file,'*.ply')):
        print(file)

        plydata = PlyData.read(file)

        num_verts = plydata['vertex'].count
        
        vertices = np.zeros(shape=[num_verts, 7], dtype=np.float32)

        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z'] #coordinates
        vertices[:,3] = plydata['vertex'].data['red']
        vertices[:,4] = plydata['vertex'].data['blue']
        vertices[:,5] = plydata['vertex'].data['green']
        vertices[:,3]=vertices[:,3]/255
        vertices[:,4]=vertices[:,4]/255
        vertices[:,5]=vertices[:,5]/255 #normalized colors
        vertices[:,6]=i #class
        i=i+1

        data=np.vstack((data,vertices[:vertices.shape[0]:int(vertices.shape[0]/6476),:]))
 

    data[:,:3]=pc_normalize(data[:,:3])

    np.random.shuffle(data)
	

    np.save('sampling/'+path_to_file.split('/')[1]+'.npy',data)


        
        
