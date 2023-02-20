import math
import os
import logging
import pyrender
import torch
import numpy as np
import trimesh
import cv2 as cv
from matplotlib import pyplot as plt
from pyhocon import ConfigFactory
from glob import glob
from tqdm import tqdm
from models.fields import NeRF, SDFNetwork, SingleVarianceNetwork, RenderingNetwork


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


def export_o_v_point_cloud(vertices):
    print(vertices.shape)
    print(np.min(vertices, axis=0))
    print(np.max(vertices, axis=0))

    res = vertices
    print(res.shape)
    with open('points.txt', 'w') as f:
        f.write(f"{vertices.shape[0]} \n")
        for i in tqdm(range(res.shape[0])):
            for j in range(res.shape[1]):
                f.write(f"{res[i][j]} ")
            f.write('\n')


def RM2Euler(RM):
    theta_z = np.arctan2(RM[1][0], RM[0][0])
    theta_y = np.arctan2(-1 * RM[2][0], np.sqrt(RM[2][1] * RM[2][1] + RM[2][2] * RM[2][2]))
    theta_x = np.arctan2(RM[2][1], RM[2][2])
    # print(f"Euler angles:\ntheta_x: {theta_x}\ntheta_y: {theta_y}\ntheta_z: {theta_z}")
    return np.array([theta_x, theta_y, theta_z])


def RM2EulerDeg(RM):
    theta_z = np.arctan2(RM[1][0], RM[0][0]) / np.pi * 180
    theta_y = np.arctan2(-1 * RM[2][0], np.sqrt(RM[2][1] * RM[2][1] + RM[2][2] * RM[2][2])) / np.pi * 180
    theta_x = np.arctan2(RM[2][1], RM[2][2]) / np.pi * 180
    # print(f"Euler angles:\ntheta_x: {theta_x}\ntheta_y: {theta_y}\ntheta_z: {theta_z}")
    return np.array([theta_x, theta_y, theta_z])


def euler2RM(theta):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta[0]), -np.sin(theta[0])],
                    [0, np.sin(theta[0]), np.cos(theta[0])]
                    ])

    R_y = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
                    [0, 1, 0],
                    [-np.sin(theta[1]), 0, np.cos(theta[1])]
                    ])

    R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                    [np.sin(theta[2]), np.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))
    # print(f"Rotate matrix:\n{R}")
    return R
# https://blog.csdn.net/weixin_41010198/article/details/115960331


class Runner:
    def __init__(self, case_name):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = 'data/womask.conf'
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case_name)
        f.close()
        self.conf = ConfigFactory.parse_string(conf_text)

        # Training parameters
        self.learning_rate = self.conf.get_float('train.learning_rate')

        # Networks
        params_to_train = []
        self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
        params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())
        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

    def load_camera_params(self):
        camera_dict = np.load('data/cameras_sphere.npz')
        n_images = len(os.listdir('data/imgs'))

        # world_mat is a projection matrix from world to image
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
        self.intrinsics_all = []
        self.pose_all = []
        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(intrinsics)
            self.pose_all.append(pose)
        self.intrinsics_all = np.stack(self.intrinsics_all)  # [n_images, 4, 4]
        self.intrinsics_all_inv = np.linalg.inv(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = np.stack(self.pose_all)  # [n_images, 4, 4]

        images_lis = sorted(glob(os.path.join('data/imgs', '*.png')))
        images_np = np.stack([cv.imread(im_name) for im_name in tqdm(images_lis)]) / 256.0
        self.images = images_np.astype(np.float32)  # [n_images, H, W, 3]
        masks_lis = sorted(glob(os.path.join('data/masks', '*.png')))
        masks_np = np.stack([cv.imread(im_name) for im_name in tqdm(masks_lis)]) / 256.0
        self.masks = masks_np.astype(np.float32)  # [n_images, H, W, 3]
        self.H, self.W = self.images.shape[1], self.images.shape[2]

    def load_ckpt(self):
        latest_model_name = None
        model_list_raw = os.listdir('data/ckpts')
        model_list = []
        for model_name in model_list_raw:
            if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= 999999:
                model_list.append(model_name)
        model_list.sort()

        if (len(model_list) > 0):
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            print(f'Find checkpoint: {latest_model_name}')

            checkpoint = torch.load(os.path.join('data/ckpts', latest_model_name), map_location=self.device)
            self.nerf_outside.load_state_dict(checkpoint['nerf'])
            self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
            self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
            self.color_network.load_state_dict(checkpoint['color_network_fine'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.iter_step = checkpoint['iter_step']
        else:
            logging.error('No checkpoint found')
            exit(-1)

    def load_mesh(self):
        latest_model_name = None
        model_list_raw = os.listdir('data/meshes')
        model_list = []
        for model_name in model_list_raw:
            if model_name[-3:] == 'ply' and int(model_name[:-4]) <= 99999999:
                model_list.append(model_name)
        model_list.sort()

        if (len(model_list) > 0):
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            print(f'Find 3d model: {latest_model_name}')

            self.mesh = trimesh.load(os.path.join('data/meshes', latest_model_name))
            self.mesh.vertices = (self.mesh.vertices - self.scale_mats_np[0][:3, 3][None]) / self.scale_mats_np[0][0, 0]
            self.mesh.invert()

            export_o_v_point_cloud(self.mesh.vertices)
        else:
            logging.error('No 3d model found')
            exit(-1)


if (__name__ == "__main__"):
    # logging.basicConfig(level=logging.DEBUG)

    runner = Runner('haibao')
    runner.load_camera_params()
    runner.load_mesh()
    # print(np.min(runner.mesh.vertices, axis=0))
    # print(np.max(runner.mesh.vertices, axis=0))

    mesh = pyrender.Mesh.from_trimesh(runner.mesh, smooth=False)
    scene = pyrender.Scene(ambient_light=[0, 0, 0], bg_color=[0, 0, 0])
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=2e3)

    id = 0
    scene.add(mesh, pose=np.eye(4))
    light_pose = np.copy(runner.pose_all[id])
    camera_pose = np.copy(runner.pose_all[id])
    light_node = scene.add(light)
    scene.set_pose(light_node, pose=light_pose)
    camera_pose[:3, :3] *= -1
    camera_node = scene.add(camera)
    scene.set_pose(camera_node, pose=camera_pose)

    # render scene
    r = pyrender.OffscreenRenderer(runner.W, runner.H)
    color, _ = r.render(scene)

    plt.figure(figsize=(8, 8))
    plt.imshow(color)
    plt.show()

    start = np.array([0, 0, 0])
    end = RM2EulerDeg(light_pose)
    os.makedirs('data/renderings', exist_ok=True)
    for i in tqdm(range(30)):
        light_pose = np.copy(runner.pose_all[i])
        camera_pose = np.copy(runner.pose_all[i])
        scene.set_pose(light_node, pose=light_pose)
        camera_pose[:3, :3] *= -1
        scene.set_pose(camera_node, pose=camera_pose)

        # render scene
        r = pyrender.OffscreenRenderer(runner.W, runner.H)
        color, _ = r.render(scene)

        plt.figure(figsize=(8, 8))
        plt.imshow(color)
        plt.savefig(os.path.join('data/renderings', '%03d.png' % i))
