import pickle
import open3d as o3d
import trimesh
import numpy as np

def render_point_cloud(path):
    print("Load a ply point cloud, print it, and render it")
    pcd = o3d.io.read_point_cloud(path)
    print(dir(pcd))
    points = np.asarray(pcd.points)
    print(points)
    print(np.max(points, axis=0))
    print(np.min(points, axis=0))
    o3d.visualization.draw_geometries([pcd])

def render_mesh(path):
    print("Testing mesh in Open3D...")
    mesh = o3d.io.read_triangle_mesh(path)
    print(mesh)
    print("Computing normal and rendering it.")
    mesh.compute_vertex_normals()
    print(np.asarray(mesh.triangle_normals))
    o3d.visualization.draw_geometries([mesh])

def load_npz(path):
    print("Load an npz file and print it")
    npz_point_cloud = np.load(path)
    for key in npz_point_cloud:
        print(key)
        print(npz_point_cloud[key].shape)

def load_pickle(path):
    f = open(path,'rb')
    data = pickle.load(f)
    print(data)

if (__name__=="__main__"):
    # e.g. path = 'outputs/mvsnet001_l3.ply'
    # ply_path = 'data/demo/Matterport3D_processed/17DRP5sb8fy/pointcloud.ply'
    # npz_path = 'data/demo/Matterport3D_processed/17DRP5sb8fy/pointcloud.npz'
    ply_path = 'out/00300000.ply'
    npz_path = 'out/cameras_sphere.npz'
    msh_path = "out/dog/00300000.ply"
    pkl_path = './out/demo_matterport/generation/time_generation_full.pkl'
    # render_point_cloud(ply_path)
    load_npz(npz_path)
    # render_mesh(msh_path)

    '''
    demo: 400000 189
    mvs1: 3739355 19444250
    mvs33: 2349185
    '''