import os
import logging
import torch
import numpy as np
import trimesh
import cv2 as cv
from pyhocon import ConfigFactory
from tqdm import tqdm

from utils import load_K_Rt_from_P
from models.fields import NeRF, SDFNetwork, SingleVarianceNetwork, RenderingNetwork
from models.renderer import NeuSRenderer


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
    def __init__(self, case_name, n_images, W, H, conf_path='womask.conf'):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.device = torch.device('cuda')
        torch.cuda.set_device(0)

        # Configuration
        self.conf_path = conf_path
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
        self.renderer = NeuSRenderer(self.nerf_outside,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     **self.conf['model.neus_renderer'])

        self.n_images = n_images
        self.W = W
        self.H = H

    def load_camera_params(self, path):
        """

        :param path: npz file
        :return:
        """
        camera_dict = np.load(path)

        # world_mat is a projection matrix from world to image
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        self.intrinsics_all = []
        self.pose_all = []
        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())
        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)  # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]

    def load_ckpt(self, path):
        print(f'Find checkpoint: {path}')

        checkpoint = torch.load(path, map_location=self.device)
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).to(self.device)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3],
                         p[:, :, :, None]).squeeze()  # 与内参矩阵相乘，W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # 求L2范式，W, H, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def near_far_from_sphere(self, rays_o, rays_d):  # https://github.com/Totoro97/NeuS/issues/11
        a = torch.sum(rays_d ** 2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def validate_image(self, idx=-1, resolution_level=1, save_path='.'):
        idxs = []
        if idx == -1:
            for idx in range(self.n_images):
                idxs.append(idx)
        else:
            idxs.append(idx)

        for idx in idxs:
            print(f'Validate camera: {idx}')

            rays_o, rays_d = self.gen_rays_at(idx, resolution_level=resolution_level)
            H, W, _ = rays_o.shape
            rays_o = rays_o.reshape(-1, 3).split(self.conf['train.batch_size'])
            rays_d = rays_d.reshape(-1, 3).split(self.conf['train.batch_size'])

            out_rgb_fine = []
            for i in tqdm(range(len(rays_o))):
                rays_o_batch = rays_o[i]
                rays_d_batch = rays_d[i]
                near, far = self.near_far_from_sphere(rays_o_batch, rays_d_batch)
                background_rgb = torch.ones([1, 3]) if self.conf['train.use_white_bkgd'] else None
                render_out = self.renderer.render(rays_o_batch,
                                                  rays_d_batch,
                                                  near,
                                                  far,
                                                  cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                                  background_rgb=background_rgb)
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
                del render_out

            img_fine = None
            if len(out_rgb_fine) > 0:
                img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)
            for i in range(img_fine.shape[-1]):
                if len(out_rgb_fine) > 0:
                    cv.imwrite(os.path.join(save_path, 'image', '{:0>3d}.png'.format(idx)), img_fine[..., i])

    def get_cos_anneal_ratio(self):
        if self.conf['train.anneal_end'] == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.conf['train.anneal_end']])


# if (__name__ == "__main__"):
#     # logging.basicConfig(level=logging.DEBUG)
#
#     runner = Runner('haibao')
#     runner.load_camera_params()
#     runner.load_mesh()
#     # print(np.min(runner.mesh.vertices, axis=0))
#     # print(np.max(runner.mesh.vertices, axis=0))
#
#     mesh = pyrender.Mesh.from_trimesh(runner.mesh, smooth=False)
#     scene = pyrender.Scene(ambient_light=[0, 0, 0], bg_color=[0, 0, 0])
#     camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
#     light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=2e3)
#
#     id = 0
#     scene.add(mesh, pose=np.eye(4))
#     light_pose = np.copy(runner.pose_all[id])
#     camera_pose = np.copy(runner.pose_all[id])
#     light_node = scene.add(light)
#     scene.set_pose(light_node, pose=light_pose)
#     camera_pose[:3, :3] *= -1
#     camera_node = scene.add(camera)
#     scene.set_pose(camera_node, pose=camera_pose)
#
#     # render scene
#     r = pyrender.OffscreenRenderer(runner.W, runner.H)
#     color, _ = r.render(scene)
#
#     plt.figure(figsize=(8, 8))
#     plt.imshow(color)
#     plt.show()
#
#     start = np.array([0, 0, 0])
#     end = RM2EulerDeg(light_pose)
#     os.makedirs('old_data/data/renderings', exist_ok=True)
#     for i in tqdm(range(30)):
#         light_pose = np.copy(runner.pose_all[i])
#         camera_pose = np.copy(runner.pose_all[i])
#         scene.set_pose(light_node, pose=light_pose)
#         camera_pose[:3, :3] *= -1
#         scene.set_pose(camera_node, pose=camera_pose)
#
#         # render scene
#         r = pyrender.OffscreenRenderer(runner.W, runner.H)
#         color, _ = r.render(scene)
#
#         plt.figure(figsize=(8, 8))
#         plt.imshow(color)
#         plt.savefig(os.path.join('old_data/data/renderings', '%03d.png' % i))
