import open3d as o3d # pip install open3d
import argparse
import rerun as rr
import numpy as np

# Argument parser for input file
parser = argparse.ArgumentParser(description="Open and visualize a PCD file using Open3D.")
parser.add_argument("--input_file", type=str, required=True, help="Path to the input PCD file.")
args = parser.parse_args()

# Read the PCD file
pcd = o3d.io.read_point_cloud(args.input_file)

# Check if the PCD file is loaded successfully
if not pcd.has_points():
    raise ValueError(f"Failed to load point cloud from {args.input_file}")

# Print some basic information about the point cloud
print("Point Cloud Information:")
print(f"Number of points: {len(pcd.points)}")

pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Visualize the point cloud
# o3d.visualization.draw_geometries([pcd])

# rerun viz
# normals = np.asarray(pcd.normals)
# # normals are in [-1, 1], remap to [0, 255]
# colors = ((normals + 1) / 2 * 255).astype(np.uint8)

rr.init("point_cloud", spawn=True)
rr.log("point_cloud", rr.Points3D(np.asarray(pcd.points)))
# rr.log("point_cloud", rr.Points3D(np.asarray(pcd.points), colors=colors))