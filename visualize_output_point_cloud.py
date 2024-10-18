import open3d as o3d

print("Load a ply point cloud, print it, and render it")
#pcd_splats = o3d.io.read_point_cloud("/root/code/output/tum/rgbd_dataset_freiburg1_desk/rgbd_dataset_freiburg1_desk_global_splats.ply")
pcd_clean = o3d.io.read_point_cloud("/root/code/output/tum/rgbd_dataset_freiburg1_desk/mesh/cleaned_mesh.ply")
o3d.visualization.draw_geometries([pcd_clean],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])