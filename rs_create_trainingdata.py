'''
Funktion zum erstellen der Traningsdaten:

1. features x = Speichere das Tiefenbild als numpy array

2. labels y = Speichere die Ausrichtung als numpy array
'''

#mendeley username marcel:

import math
import ctypes
import pyglet
import pyglet.gl as gl
import numpy as np
import pyrealsense2 as rs
import cv2
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Visualize the test data
import scipy.io as io
import scipy.ndimage as nd

from pyntcloud import PyntCloud # open source library for 3D pointcloud visualisation

from plyfile import PlyData, PlyElement
import os

'''

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Get stream profile and camera intrinsics
profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
w, h = depth_intrinsics.width, depth_intrinsics.height

# Processing blocks
pc = rs.pointcloud()
decimate = rs.decimation_filter()
decimate.set_option(rs.option.filter_magnitude, 2 ** 0)
colorizer = rs.colorizer()



try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_frame = decimate.process(depth_frame)

        # Grab new intrinsics (may be changed by decimation)
        depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
        w, h = depth_intrinsics.width, depth_intrinsics.height

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # ---------------------------------------- Filtering the Depth Stream ------------------------------------------

        # Spartial Filter
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.filter_magnitude, 5)
        spatial.set_option(rs.option.filter_smooth_alpha, 1)
        spatial.set_option(rs.option.filter_smooth_delta, 50)

        # Basic hole filling Filter
        spatial.set_option(rs.option.holes_fill, 3)
        hole_filling = rs.hole_filling_filter()

        filtered_depth = spatial.process(depth_frame)
        filtered_depth = hole_filling.process(depth_frame)

        depth_colormap = np.asanyarray(colorizer.colorize(filtered_depth).get_data())

        points = pc.calculate(filtered_depth)
        pc.map_to(color_frame)

        # Pointcloud data to arrays
        v, t = points.get_vertices(), points.get_texture_coordinates()
        verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
        texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv




        # ---------------------------------------> Visualization ------------------------------------------------------



        # Konvertieren der Bilddaten in numpy Arrays
        #depth_image = np.asanyarray(depth_frame.get_data())  # depth_image[y][x]
        #color_image = np.asanyarray(color_frame.get_data())

        #pc.map_to(color_frame)
        #pointcloud = pc.calculate(depth_frame)
        

        # -------------------- Save Depthimage-----------------------------

        from datetime import datetime
        now = datetime.now()  # current date and time
        date_time = now.strftime("%d-%m_%H-%M-%S")

        key = cv2.waitKey(1)

        if key & 0xFF == ord('s') or key == 115:

            print("Saving ")

            points.export_to_ply('Saved_Depth_Data/Saved_Depth_Array' + date_time + ".ply", color_frame)
            np.save('Saved_Depth_Data/Saved_Depth_Array' + date_time, verts)

            cv2.imwrite("Saved_Depth_Data/ColorOut" + date_time + ".png", color_image)

            print("Saved_Depth_Data/Depth saved", date_time)

        # Ausgabe

        # Render images
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        cv2.namedWindow('Save Depth Data', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Save Depth Data', depth_colormap)

        #key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()


'''

# ------------------------------------------------- Testing ------------------------------------------------------------

DATADIR = 'Saved_Depth_Data'


#depth_array = os.path.join(DATADIR, depthimg)

# Laden der mit RS erstellen Depth Image
cloud = PyntCloud.from_file("Saved_Depth_Data/Saved_Depth_Array01-04_18-54-20.ply")

xd = np.load("Saved_Depth_Data/Saved_Depth_Array01-04_18-19-16.npy")

# Erstellen eines Voxelgrids
voxelgrid_id = cloud.add_structure("voxelgrid", n_x=64, n_y=64, n_z=64)

voxelgrid = cloud.structures[voxelgrid_id]

voxelgrid.plot(d=3, mode="density", cmap="hsv")


# Transformieren des Voxelgrids zu binary 3D Array (1 = Punkt, 0= nichts)
binary_feature_vector = voxelgrid.get_feature_vector(mode="binary")



#Plotten der Depth Images
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect('equal')
ax.voxels(binary_feature_vector, edgecolor='red')
plt.savefig("Saved_Depth_Data/PlottedRSData.png")



