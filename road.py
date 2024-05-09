import open3d as o3d
import numpy as np
import os

MESH_FILE="resources/kota_circuit2.ply"

# Check if the file exists
if not os.path.exists(MESH_FILE):
    print(f'File "{MESH_FILE}" not found')
    exit()

# Read the point cloud from the `.ply` file
ply = o3d.io.read_point_cloud(MESH_FILE)

# Remove forest and mountain from the point cloud
# Define the region of interest
center = ply.get_center()
radius = 200
min_bound=(center[0]-radius+20, center[1]-radius+80, -100) # FIXED
max_bound=(center[0]+radius+20, center[1]+radius+80, -88)

bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
cropped = ply.crop(bbox)

# Draw the bounding box with red wireframe
bbox = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bbox)
bbox.paint_uniform_color([1, 0, 0])

#==============================================================================

road_bounds = np.array([[0.3, 0.3, 0.3], [0.9, 0.9, 0.9]])
road_mask = np.all(
    (np.asarray(cropped.colors) >= road_bounds[0]) & (np.asarray(cropped.colors) <= road_bounds[1]),
    axis=1
)
road_pcb = cropped.select_by_index(np.where(road_mask)[0], invert=False)
mean_z = np.mean(np.asarray(road_pcb.points)[:, 2])
z_tolerance = 4
elevation_mask = np.abs(np.asarray(road_pcb.points)[:, 2] - mean_z) < z_tolerance
filtered_ply = road_pcb.select_by_index(np.where(elevation_mask)[0], invert=False)

# Remove noise using statistical outlier removal
cl, ind = filtered_ply.remove_statistical_outlier(nb_neighbors=30, std_ratio=0.5)
filtered_ply = filtered_ply.select_by_index(ind)

# Downsample the point cloud with a voxel of 1.00
down_pcd = filtered_ply.voxel_down_sample(voxel_size=1.00)

# Draw geometry and open 3D window
o3d.visualization.draw_geometries([down_pcd, bbox])