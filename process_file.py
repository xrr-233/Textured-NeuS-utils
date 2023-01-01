import os
import shutil
import numpy as np
import open3d as o3d
import argparse

BASE_PATH = os.getcwd()
MVS_PATH = "../MVSNet_pytorch/outputs"
CON_PATH = "../convolutional_occupancy_networks/data/outputs"

if (os.path.exists(CON_PATH)):
    print("CON_PATH detected, deleting...")
    shutil.rmtree(CON_PATH)
if (not os.path.exists(CON_PATH)):
    print("Create CON_PATH")
    os.mkdir(CON_PATH)
os.chdir(MVS_PATH)
folders = os.listdir()

parser = argparse.ArgumentParser(description='A tool to transfer MVSNet output to Convolutional Occupancy Network input')
parser.add_argument('--num', type=int, default=len(folders) / 2, help='Point cloud scenes generated in MVS to use in CON (should be less than max MVS scenes)')
args = parser.parse_args()

scene_num = int(args.num)
if (scene_num <= 0 and scene_num > len(folders) / 2):
    print('Invalid input!')
    exit(-1)

file_dict = []
for idx, filename in enumerate(folders):
    if (idx == scene_num):
        break
    if (filename[-4:] != ".ply"):
        file_dict.append((int(filename[4:]), filename))

for idx in range(scene_num):
    for filename in folders:
        if (filename[-4:] == ".ply" and int(filename[6:9]) == file_dict[idx][0]):
            print(f"Processing {file_dict[idx][1]}")
            src = f"{os.getcwd()}/{filename}"
            os.chdir(BASE_PATH)
            os.chdir(CON_PATH)
            os.mkdir(file_dict[idx][1])
            dst = f"{file_dict[idx][1]}/mvsnet.ply"
            shutil.copyfile(src, dst)
            pcd = o3d.io.read_point_cloud(dst)
            kwargs = {'points': np.asarray(pcd.points), 'normals': np.zeros(np.asarray(pcd.points).shape)}
            print(kwargs['normals'].shape)
            np.savez(f"{file_dict[idx][1]}/mvsnet.npz", **kwargs)
            os.chdir(BASE_PATH)
            os.chdir(MVS_PATH)
            break

os.chdir(BASE_PATH)
os.chdir(CON_PATH)
with open('test.lst', 'w') as f:
    for idx in range(scene_num):
        f.write(f"{file_dict[idx][1]}\n")