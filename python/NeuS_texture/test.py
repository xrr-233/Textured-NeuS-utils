import copy
import logging
import math
import os
import shutil
import cv2
import lpips
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from runner import Runner
from utils import load_K_Rt_from_P


class Dataset:
    def __init__(self):
        """
        applicable only for ONE SINGLE SOLE model
        """
        self.n_images = 0
        self.W = 0
        self.H = 0
        self.camera_intrinsics = []
        self.camera_extrinsics = []
        self.meshes = []

    def load_camera_parameters(self, src_root, convert=None):
        """
        指定某一个模型，导入并储存其相机参数，用以渲染和生成camera_sphere.npz

        :param src_root: 模型主文件夹路径
        :param convert: 是否需要变换，如是则输入4*4矩阵
        :return:
        """
        camera_images = os.listdir(os.path.join(src_root, 'image'))
        self.n_images = len(camera_images)
        sample_img = Image.open(os.path.join(src_root, 'image', camera_images[0]))
        self.W = sample_img.width
        self.H = sample_img.height

        camera_dict = np.load(os.path.join(src_root, 'cameras_sphere.npz'))
        world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        if convert is not None:
            convert_mat = np.array(convert, dtype=np.float32)
        else:
            convert_mat = np.eye(4, dtype=np.float32)
        self.camera_intrinsics = []
        self.camera_extrinsics = []
        for scale_mat, world_mat in zip(scale_mats_np, world_mats_np):
            P = world_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.camera_intrinsics.append(intrinsics[:3, :3])
            if convert:
                pose = pose @ convert_mat
            self.camera_extrinsics.append(pose)

    def generate_baseline_rendered_mesh(self, mesh_show_back_face=True, save_path='.'):
        """渲染单个模型

        必须在指定内外参（即运行load_camera_parameters）后执行，将该模型以指定的内外参进行图像渲染

        :param mesh_show_back_face: 渲染的时候法线背面要不要透
        :param save_path: 保存主文件夹路径
        :return:
        """
        trajectory = []
        for i in range(self.n_images):
            pinhole_parameters = o3d.camera.PinholeCameraParameters()
            pinhole_intrinsic = o3d.camera.PinholeCameraIntrinsic()
            pinhole_intrinsic.intrinsic_matrix = self.camera_intrinsics[i]
            pinhole_parameters.intrinsic = pinhole_intrinsic
            pinhole_parameters.extrinsic = self.camera_extrinsics[i]
            trajectory.append(pinhole_parameters)
        pinhole_trajectory = o3d.camera.PinholeCameraTrajectory()
        pinhole_trajectory.parameters = trajectory

        self.custom_draw_geometry_with_camera_trajectory = {
            'index': -1,
            'trajectory': pinhole_trajectory,
            'vis': o3d.visualization.Visualizer()
        }

        def move_forward(vis):
            """
            This function is called within the o3d.visualization.Visualizer::run() loop

            The run loop calls the function, then re-render

            So the sequence in this function is to:
            1. Capture frame
            2. index++, check ending criteria
            3. Set camera
            4. (Re-render)
            """
            ctr = vis.get_view_control()
            glb = self.custom_draw_geometry_with_camera_trajectory
            if glb['index'] >= 0:
                print(os.path.join(save_path, 'image_mesh', '{:0>3d}.png'.format(glb['index'])))
                vis.capture_screen_image(os.path.join(save_path, 'image_mesh', '{:0>3d}.png'.format(glb['index'])),
                                         False)
            glb['index'] = glb['index'] + 1
            if glb['index'] < len(glb['trajectory'].parameters):
                ctr.convert_from_pinhole_camera_parameters(
                    glb['trajectory'].parameters[glb['index']], allow_arbitrary=True)
            else:
                self.custom_draw_geometry_with_camera_trajectory['vis']. \
                    register_animation_callback(None)
            return False

        vis = self.custom_draw_geometry_with_camera_trajectory['vis']
        vis.create_window(width=self.W, height=self.H)
        print(self.meshes)
        for mesh in self.meshes:
            vis.add_geometry(mesh)
        vis.get_render_option().mesh_show_back_face = mesh_show_back_face
        vis.register_animation_callback(move_forward)
        vis.run()
        vis.destroy_window()


class BlendedMVSDataset(Dataset):
    def __init__(self, path_root):
        super().__init__()
        self.path_root = 'BlendedMVS_preprocessed'
        self.image_dirs = []
        for filename in os.listdir(path_root):
            if filename.startswith('dataset_full_res_'):
                self.image_dirs.append(os.path.join(path_root, filename))
        self.textured_mesh_dir = os.path.join(path_root, 'dataset_textured_meshes')

        all_models_root = []
        all_models = []
        self.all_models_root = []
        self.all_models = []
        self.rewrite = []
        for image_dir in self.image_dirs:
            all_model = os.listdir(image_dir)
            all_models_root.extend(image_dir for _ in range(len(all_model)))
            all_models.extend(all_model)
        for filename in os.listdir(self.textured_mesh_dir):
            if filename in all_models:
                self.all_models_root.append(all_models_root[all_models.index(filename)])
                self.all_models.append(filename)
                self.rewrite.append(False)

    def __len__(self):
        return len(self.all_models)

    def get_single_model_path_root(self, identifier):
        if type(identifier) == str:
            if os.path.exists(os.path.join(self.path_root, identifier)):
                return os.path.join(self.path_root, identifier)
            else:
                print('model not found!')
                exit(-1)
        else:
            if identifier < self.__len__():
                return os.path.join(self.path_root, self.all_models[identifier])
            else:
                print('idx exceeds max model number!')
                exit(-1)

    def set_rewrite(self, identifier):
        if type(identifier) == str:
            if identifier in self.all_models:
                self.rewrite[self.all_models.index(identifier)] = True
        else:
            if identifier < self.__len__():
                self.rewrite[identifier] = True

    def rewrite_all(self):
        for i in range(len(self.rewrite)):
            self.rewrite[i] = True

    def load_single_model(self, identifier):
        """加载单个模型

        指定某一个模型，将其导入meshes序列以备之后用Open3D渲染，并初始化相关参数

        :param identifier: 指定3D模型用
        :return:
        """
        filename = None
        if type(identifier) == str:
            filename = identifier
        else:
            if identifier < self.__len__():
                filename = self.all_models[identifier]
            else:
                print('idx exceeds max model number!')
                exit(-1)

        textured_mesh_path = os.path.join(self.textured_mesh_dir, filename, 'textured_mesh')
        self.meshes = []
        for f in os.listdir(textured_mesh_path):
            if f.endswith('.obj'):
                mesh = o3d.io.read_triangle_mesh(os.path.join(textured_mesh_path, f), True)
                mesh.compute_vertex_normals()  # allow light effect
                self.meshes.append(mesh)

        self.load_camera_parameters(os.path.join(self.get_single_model_path_root(identifier)), convert=[
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ])

    def export_bundler_out(self, dst_root):
        with open(os.path.join(dst_root, 'meshlab_camera.out'), 'w') as f:
            f.write('# Bundle file v0.3\n')
            f.write(f'{self.n_images} 0\n')
            for i in range(self.n_images):
                f.write('%.6f %.6f %.6f\n' % (
                    self.camera_intrinsics[i][0, 0],
                    self.camera_intrinsics[i][0, 2],
                    self.camera_intrinsics[i][1, 2]
                ))
                f.write('%.6f %.6f %.6f\n' % (
                    self.camera_extrinsics[i][0, 0],
                    self.camera_extrinsics[i][0, 1],
                    self.camera_extrinsics[i][0, 2]
                ))
                f.write('%.6f %.6f %.6f\n' % (
                    self.camera_extrinsics[i][1, 0],
                    self.camera_extrinsics[i][1, 1],
                    self.camera_extrinsics[i][1, 2]
                ))
                f.write('%.6f %.6f %.6f\n' % (
                    self.camera_extrinsics[i][2, 0],
                    self.camera_extrinsics[i][2, 1],
                    self.camera_extrinsics[i][2, 2]
                ))
                f.write('%.6f %.6f %.6f\n' % (
                    self.camera_extrinsics[i][0, 3],
                    self.camera_extrinsics[i][1, 3],
                    self.camera_extrinsics[i][2, 3]
                ))

    def preprocess_dataset(self, debug=False):
        # device = o3d.core.Device("CPU:0")
        os.makedirs(self.path_root, exist_ok=True)

        for i in tqdm(range(self.__len__())):
            filename_root = self.all_models_root[i]
            filename = self.all_models[i]

            if self.rewrite[i]:
                if os.path.exists(f'{self.path_root}/{filename}'):
                    shutil.rmtree(f'{self.path_root}/{filename}')
            if not os.path.exists(f'{self.path_root}/{filename}'):
                os.makedirs(f'{self.path_root}/{filename}', exist_ok=True)
                os.makedirs(f'{self.path_root}/{filename}/image', exist_ok=True)
                os.makedirs(f'{self.path_root}/{filename}/image_mesh', exist_ok=True)
                os.makedirs(f'{self.path_root}/{filename}/mask', exist_ok=True)

                # region Process image
                src_root = os.path.join(filename_root, filename, filename, filename, 'blended_images')
                dst_root = f'{self.path_root}/{filename}'
                for index, file in enumerate(os.listdir(src_root)):
                    img = cv2.imread(os.path.join(src_root, file))
                    cv2.imwrite(os.path.join(dst_root, 'image', '{:0>3d}.png'.format(index)), img)
                    cv2.imwrite(os.path.join(dst_root, 'mask', '{:0>3d}.png'.format(index)), np.ones_like(img) * 255)
                # endregion

                # region Process camera_sphere.npz
                src_root = os.path.join(filename_root, filename, filename, filename, 'cams')
                cam_dict = dict()
                convert_mat = np.zeros([4, 4], dtype=np.float32)
                convert_mat[0, 1] = 1.0
                convert_mat[1, 0] = 1.0
                convert_mat[2, 2] = -1.0
                convert_mat[3, 3] = 1.0
                self.camera_extrinsics = []
                self.camera_intrinsics = []
                for index, file in enumerate(os.listdir(src_root)):
                    if file == 'pair.txt':
                        continue
                    with open(os.path.join(src_root, file), 'r') as f:
                        all_lines = f.readlines()
                        row_1 = all_lines[1].split(' ')[:4]
                        row_2 = all_lines[2].split(' ')[:4]
                        row_3 = all_lines[3].split(' ')[:4]
                        row_4 = all_lines[4].split(' ')[:4]
                        extrinsic = np.array([row_1, row_2, row_3, row_4], dtype=np.float32)
                        self.camera_extrinsics.append(extrinsic)
                        extrinsic = extrinsic @ convert_mat
                        row_1 = all_lines[7].split(' ')[:3]
                        row_1.append('0')
                        row_2 = all_lines[8].split(' ')[:3]
                        row_2.append('0')
                        row_3 = all_lines[9].split(' ')[:3]
                        row_3.append('0')
                        row_4 = ['0', '0', '0', '1']
                        intrinsic = np.array([row_1, row_2, row_3, row_4], dtype=np.float32)
                        self.camera_intrinsics.append(intrinsic[:3, :3])
                    w2c = np.linalg.inv(extrinsic)
                    world_mat = intrinsic @ w2c
                    world_mat = world_mat.astype(np.float32)
                    cam_dict['camera_mat_{}'.format(index)] = intrinsic
                    cam_dict['camera_mat_inv_{}'.format(index)] = np.linalg.inv(intrinsic)
                    cam_dict['world_mat_{}'.format(index)] = world_mat
                    cam_dict['world_mat_inv_{}'.format(index)] = np.linalg.inv(world_mat)

                src_root = os.path.join(self.textured_mesh_dir, filename, 'textured_mesh')
                vertices = []
                for f in os.listdir(src_root):
                    if f.endswith('.obj'):
                        mesh = o3d.io.read_triangle_mesh(os.path.join(src_root, f), True)
                        # mesh.compute_vertex_normals()  # allow light effect
                        vertices.append(np.asarray(mesh.vertices))
                vertices = np.concatenate(vertices, axis=0)
                bbox_max = np.max(vertices, axis=0)
                bbox_min = np.min(vertices, axis=0)
                center = (bbox_max + bbox_min) * 0.5
                radius = np.linalg.norm(vertices - center, ord=2, axis=-1).max()
                scale_mat = np.diag([radius, radius, radius, 1.0]).astype(np.float32)
                scale_mat[:3, 3] = center
                self.n_images = len(self.camera_extrinsics)
                for i in range(self.n_images):
                    cam_dict['scale_mat_{}'.format(i)] = scale_mat
                    cam_dict['scale_mat_inv_{}'.format(i)] = np.linalg.inv(scale_mat)
                np.savez(os.path.join(dst_root, 'cameras_sphere.npz'), **cam_dict)
                # endregion

                # region Generate bundler .out
                self.export_bundler_out(dst_root)
                # endregion

                # region Process image_mesh
                self.load_single_model(filename)
                self.generate_baseline_rendered_mesh(save_path=dst_root)
                # endregion

                # region Print sampled mesh vertices as well as camera 3D position (debug=True)
                if debug:
                    with open(os.path.join(dst_root, 'points2.txt'), 'w') as f:
                        for i in range(self.n_images):
                            f.write(
                                f'{self.camera_extrinsics[i][0, 3]} {self.camera_extrinsics[i][1, 3]} {self.camera_extrinsics[i][2, 3]}\n')
                        indices = np.linspace(0, len(vertices), 10000, dtype=int)
                        print(indices[:10])
                        for i in indices:
                            if (i == len(vertices)):
                                f.write(f'{vertices[i - 1][0]} {vertices[i - 1][1]} {vertices[i - 1][2]}\n')
                            else:
                                f.write(f'{vertices[i][0]} {vertices[i][1]} {vertices[i][2]}\n')
                else:
                    if os.path.exists(os.path.join(dst_root, 'points2.txt')):
                        os.remove(os.path.join(dst_root, 'points2.txt'))
                # endregion


class DTUDataset(Dataset):
    def __init__(self, path_root):
        super().__init__()
        self.path_root = 'DTUDataset_preprocessed'
        self.all_models_root = os.path.join(path_root, './SampleSet/MVS Data')
        self.all_models = []
        self.rewrite = []

        for folder in os.listdir(os.path.join(self.all_models_root, 'Rectified')):
            self.all_models.append(folder)
            self.rewrite.append(False)

    def __len__(self):
        return len(self.all_models)

    def get_single_model_path_root(self, identifier):
        if type(identifier) == str:
            if os.path.exists(os.path.join(self.path_root, identifier)):
                return os.path.join(self.path_root, identifier)
            else:
                print('model not found!')
                exit(-1)
        else:
            if identifier < self.__len__():
                return os.path.join(self.path_root, self.all_models[identifier])
            else:
                print('idx exceeds max model number!')
                exit(-1)

    def set_rewrite(self, identifier):
        if type(identifier) == str:
            if identifier in self.all_models:
                self.rewrite[self.all_models.index(identifier)] = True
        else:
            if identifier < self.__len__():
                self.rewrite[identifier] = True

    def rewrite_all(self):
        for i in range(len(self.rewrite)):
            self.rewrite[i] = True

    def load_single_model(self, identifier):
        """加载单个模型

        指定某一个模型，将其导入meshes序列以备之后用Open3D渲染，并初始化相关参数

        :param identifier: 指定3D模型用
        :return:
        """
        filename = None
        if type(identifier) == str:
            filename = identifier
        else:
            if identifier < self.__len__():
                filename = self.all_models[identifier]
            else:
                print('idx exceeds max model number!')
                exit(-1)

        self.meshes = [o3d.io.read_triangle_mesh(os.path.join(self.path_root, filename, 'textured_mesh.ply'), True)]
        self.meshes[0].compute_vertex_normals()  # allow light effect
        self.load_camera_parameters(os.path.join(self.get_single_model_path_root(identifier)))

    def preprocess_dataset(self):
        os.makedirs(self.path_root, exist_ok=True)

        for i in tqdm(range(self.__len__())):
            filename = self.all_models[i]

            if self.rewrite[i]:
                if os.path.exists(f'{self.path_root}/{filename}'):
                    shutil.rmtree(f'{self.path_root}/{filename}')
            if not os.path.exists(f'{self.path_root}/{filename}'):
                os.makedirs(f'{self.path_root}/{filename}', exist_ok=True)
                os.makedirs(f'{self.path_root}/{filename}/image', exist_ok=True)
                os.makedirs(f'{self.path_root}/{filename}/image_mesh', exist_ok=True)
                os.makedirs(f'{self.path_root}/{filename}/mask', exist_ok=True)

                # region Process image
                src_root = os.path.join(self.all_models_root, 'Rectified', filename)
                dst_root = f'{self.path_root}/{filename}'
                index = 0
                for file in os.listdir(src_root):
                    if file.endswith('max.png'):
                        img = cv2.imread(os.path.join(src_root, file))
                        cv2.imwrite(os.path.join(dst_root, 'image', '{:0>3d}.png'.format(index)), img)
                        cv2.imwrite(os.path.join(dst_root, 'mask', '{:0>3d}.png'.format(index)), np.ones_like(img) * 255)
                        index += 1
                # endregion

                # region Process camera_sphere.npz
                src_root = os.path.join(self.all_models_root, 'Calibration', 'cal18')
                cam_dict = dict()
                self.camera_extrinsics = []
                self.camera_intrinsics = []
                index = 0
                for file in os.listdir(src_root):
                    if file.endswith('.txt'):
                        world_mat = np.zeros((3, 4), dtype=np.float32)
                        with open(os.path.join(src_root, file), 'r') as f:
                            all_lines = f.readlines()
                            for i, line in enumerate(all_lines):
                                elements = line.split(' ')
                                if len(elements) > 0:
                                    for j in range(4):
                                        world_mat[i, j] = float(elements[j])
                            intrinsic, pose = load_K_Rt_from_P(None, world_mat)
                            self.camera_extrinsics.append(pose)
                            self.camera_intrinsics.append(intrinsic[:3, :3])
                        world_mat = np.concatenate([world_mat, np.array([0, 0, 0, 1]).reshape(1, 4)], axis=0)
                        cam_dict['camera_mat_{}'.format(index)] = intrinsic
                        cam_dict['camera_mat_inv_{}'.format(index)] = np.linalg.inv(intrinsic)
                        cam_dict['world_mat_{}'.format(index)] = world_mat
                        cam_dict['world_mat_inv_{}'.format(index)] = np.linalg.inv(world_mat)
                        index += 1

                src_root = os.path.join(self.all_models_root, 'Surfaces', 'camp')
                vertices = []
                for f in os.listdir(src_root):
                    if int(f[4:7]) == int(filename[4:]):
                        shutil.copy(os.path.join(src_root, f),
                                    os.path.join(self.path_root, filename, 'textured_mesh.ply'))
                        mesh = o3d.io.read_triangle_mesh(os.path.join(src_root, f), True)
                        mesh.compute_vertex_normals()  # allow light effect
                        vertices.append(np.asarray(mesh.vertices))
                vertices = np.concatenate(vertices, axis=0)
                bbox_max = np.max(vertices, axis=0)
                bbox_min = np.min(vertices, axis=0)
                center = (bbox_max + bbox_min) * 0.5
                radius = np.linalg.norm(vertices - center, ord=2, axis=-1).max()
                scale_mat = np.diag([radius, radius, radius, 1.0]).astype(np.float32)
                scale_mat[:3, 3] = center
                self.n_images = len(self.camera_extrinsics)
                for i in range(self.n_images):
                    cam_dict['scale_mat_{}'.format(i)] = scale_mat
                    cam_dict['scale_mat_inv_{}'.format(i)] = np.linalg.inv(scale_mat)
                np.savez(os.path.join(dst_root, 'cameras_sphere.npz'), **cam_dict)
                # endregion

                # region Process image_mesh
                self.load_single_model(filename)
                self.generate_baseline_rendered_mesh(save_path=dst_root)
                # endregion


class SelfDataset(Dataset):
    def __init__(self, path_public, path_exp):
        super().__init__()
        self.path_root = 'SelfDataset_preprocessed'
        self.path_public = path_public
        self.path_exp = path_exp

    def load_model(self):
        self.meshes = [o3d.io.read_triangle_mesh(os.path.join(self.path_root, 'textured_mesh.ply'), True)]

    def preprocess_dataset(self, rewrite=False):
        if rewrite:
            shutil.rmtree(self.path_root)
        if not os.path.exists(self.path_root):
            os.makedirs(self.path_root, exist_ok=True)
            shutil.copytree(os.path.join(self.path_public, 'image'),
                            os.path.join(self.path_root, 'image'))
            shutil.copytree(os.path.join(self.path_public, 'mask'),
                            os.path.join(self.path_root, 'mask'))
            shutil.copy(os.path.join(self.path_public, 'cameras_sphere.npz'),
                        os.path.join(self.path_root, 'cameras_sphere.npz'))
            shutil.copy(os.path.join(self.path_exp, 'womask_sphere', 'meshes', 'vertex_color.ply'),
                        os.path.join(self.path_root, 'textured_mesh.ply'))

            # region Process mesh
            os.makedirs(os.path.join(self.path_root, 'image_mesh'), exist_ok=True)
            self.load_model()
            self.generate_baseline_rendered_mesh(save_path=self.path_root)
            # endregion
        self.load_camera_parameters(self.path_root)


class TexturedNeUSDataset(Dataset):
    def __init__(self, path_root):
        super().__init__()
        self.path_root = 'TexturedNeUSDataset_processed'
        self.all_models_root = path_root

    def get_single_model_path_root(self, identifier):
        if os.path.exists(os.path.join(self.path_root, identifier)):
            return os.path.join(self.path_root, identifier)
        else:
            print('model not found!')
            exit(-1)

    def load_model(self, identifier):
        self.meshes = [o3d.io.read_triangle_mesh(os.path.join(self.path_root, identifier, 'textured_mesh.ply'), True)]

    def process_dataset(self, identifier, rewrite=False):
        """

        :param identifier: can only be type(str), int is not applicable
        :param rewrite:
        :return:
        """
        path_public = os.path.join(self.all_models_root, 'public_data', identifier)
        path_exp = os.path.join(self.all_models_root, 'exp', identifier)
        if rewrite and os.path.exists(os.path.join(self.path_root, identifier)):
            shutil.rmtree(os.path.join(self.path_root, identifier))
        if not os.path.exists(os.path.join(self.path_root, identifier)) and os.path.exists(path_public):
            os.makedirs(os.path.join(self.path_root, identifier), exist_ok=True)

            # region Process checkpoint
            model_list_raw = os.listdir(os.path.join(path_exp, 'womask_sphere', 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth':
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]
            shutil.copy(os.path.join(path_exp, 'womask_sphere', 'checkpoints', latest_model_name),
                        os.path.join(self.path_root, identifier, 'ckpt.pth'))
            # endregion

            # region Process cameras_sphere.npz
            shutil.copy(os.path.join(path_public, 'cameras_sphere.npz'),
                        os.path.join(self.path_root, identifier, 'cameras_sphere.npz'))
            # endregion

            # region Process image
            # 由于在本机渲染过于慢，请手动在高配GPU服务器上高batch训练
            os.makedirs(os.path.join(self.path_root, identifier, 'image'), exist_ok=True)
            self.load_camera_parameters(path_public)
            '''
            self.runner = Runner('', self.n_images, self.W, self.H)
            self.runner.load_camera_params(os.path.join(self.path_root, 'cameras_sphere.npz'))
            self.runner.load_ckpt(os.path.join(self.path_root, 'ckpt.pth'))
            self.runner.validate_image(save_path=self.path_root)
            '''
            # endregion

            # region Process mesh
            shutil.copy(os.path.join(path_exp, 'womask_sphere', 'meshes', 'vertex_color.ply'),
                        os.path.join(self.path_root, identifier, 'textured_mesh.ply'))
            os.makedirs(os.path.join(self.path_root, identifier, 'image_mesh'), exist_ok=True)
            self.load_model(identifier)
            self.generate_baseline_rendered_mesh(save_path=os.path.join(self.path_root, identifier))
            # endregion
        self.load_camera_parameters(os.path.join(self.path_root, identifier))


class Metrics:
    def __init__(self, baseline_path, actual_path):
        self.baseline_path = baseline_path
        self.actual_path = actual_path

        camera_dict = np.load(os.path.join(baseline_path, 'cameras_sphere.npz'))
        self.scale_mat = camera_dict['scale_mat_0'].astype(np.float32)

    def check_image_dir(self, target):
        if not target == 'image' and not target == 'image_mesh':
            print('Parameter "target" only accepts "image" or "image_mesh"')
            return None, None
        src_path = os.path.join(self.baseline_path, target)
        dst_path = os.path.join(self.actual_path, target)

        flag = False
        if os.path.exists(src_path) and os.path.exists(dst_path) \
                and len(os.listdir(src_path)) == len(os.listdir(dst_path)):
            flag = True
        if not flag:
            print('Error, please check the files.')
            return None, None
        return src_path, dst_path

    def PSNR(self, target, idx=-1):
        """
        30db以上为佳

        :param target: 'image' or 'image_mesh'
        :param idx: if -1, take average
        :return:
        """
        src_path, dst_path = self.check_image_dir(target)
        if src_path == None:
            return -1

        src_file = os.listdir(src_path)
        dst_file = os.listdir(dst_path)
        idxs = []
        if idx == -1:
            for idx in range(len(os.listdir(src_path))):
                idxs.append(idx)
        else:
            idxs.append(idx)

        all_psnr = []
        for idx in tqdm(idxs):
            img1 = np.asarray(Image.open(os.path.join(src_path, src_file[idx])))
            img2 = np.asarray(Image.open(os.path.join(dst_path, dst_file[idx])))
            mse = np.mean((img1 - img2) ** 2)
            if mse == 0:
                all_psnr.append(100)
            else:
                PIXEL_MAX = 255.0
                all_psnr.append(20 * math.log10(PIXEL_MAX / math.sqrt(mse)))
        all_psnr = np.asarray(all_psnr)
        return all_psnr

    def SSIM(self, target, idx=-1):
        """
        (0, 1)越大越好

        :param target: 'image' or 'image_mesh'
        :param idx: if -1, take average
        :return:
        """
        src_path, dst_path = self.check_image_dir(target)
        if src_path == None:
            return -1

        src_file = os.listdir(src_path)
        dst_file = os.listdir(dst_path)
        idxs = []
        if idx == -1:
            for idx in range(len(os.listdir(src_path))):
                idxs.append(idx)
        else:
            idxs.append(idx)

        all_ssim = []
        for idx in tqdm(idxs):
            img1 = np.asarray(Image.open(os.path.join(src_path, src_file[idx])))
            img2 = np.asarray(Image.open(os.path.join(dst_path, dst_file[idx])))

            all_ssim.append(ssim(img1, img2, channel_axis=2))

        all_ssim = np.asarray(all_ssim)
        return all_ssim

    def LPIPS(self, target, idx=-1):
        """
        (0, 1)越小越好

        :param target: 'image' or 'image_mesh'
        :param idx: if -1, take average
        :return:
        """
        src_path, dst_path = self.check_image_dir(target)
        if src_path == None:
            return -1

        src_file = os.listdir(src_path)
        dst_file = os.listdir(dst_path)
        idxs = []
        if idx == -1:
            for idx in range(len(os.listdir(src_path))):
                idxs.append(idx)
        else:
            idxs.append(idx)

        loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores
        all_lpips = []
        for idx in tqdm(idxs):
            img1 = np.asarray(Image.open(os.path.join(src_path, src_file[idx])))
            img2 = np.asarray(Image.open(os.path.join(dst_path, dst_file[idx])))

            all_lpips.append(
                loss_fn_alex(torch.tensor(img1.transpose((2, 0, 1)), dtype=torch.float32) / 255,
                             torch.tensor(img2.transpose((2, 0, 1)), dtype=torch.float32) / 255)
                    .reshape(-1).detach().numpy()[0])

        all_lpips = np.asarray(all_lpips)
        return all_lpips

    def split_batch(self, mesh, batch_num=1024):
        """
        占用内存比较大，建议在高配机子上训

        :param mesh:
        :param batch_num:
        :return:
        """
        mesh_complete = mesh.shape[0] // batch_num
        mesh_surplus = mesh[mesh_complete * batch_num:]
        mesh = np.split(mesh[:mesh_complete * batch_num], mesh_complete)
        mesh_complete = np.broadcast_to(mesh_surplus[-1], (batch_num, 3)).copy()
        mesh_complete[:len(mesh_surplus)] = mesh_surplus
        mesh.append(mesh_complete)
        for i in range(len(mesh)):
            mesh[i] = mesh[i].reshape((mesh[i].shape[0], 1, 3))
        mesh = np.concatenate(mesh, axis=1)
        return mesh

    def ChamferL1(self):
        from ChamferDistancePytorch.chamfer_python import distChamfer
        # logging.basicConfig(level=logging.DEBUG)

        mesh_baseline = np.asarray(
            o3d.io.read_triangle_mesh(os.path.join(self.baseline_path, 'textured_mesh.ply'), True).vertices,
            dtype=np.float32)
        mesh_baseline = (mesh_baseline - self.scale_mat[:3, 3][None]) / self.scale_mat[0, 0]
        logging.debug(mesh_baseline.shape)
        mesh_baseline_len = mesh_baseline.shape[0]
        mesh_baseline = self.split_batch(mesh_baseline)
        bbox_max = np.max(mesh_baseline.reshape((-1, 3)), axis=0)
        bbox_min = np.min(mesh_baseline.reshape((-1, 3)), axis=0)
        logging.debug((bbox_max, bbox_min))
        mesh_actual = np.asarray(
            o3d.io.read_triangle_mesh(os.path.join(self.actual_path, 'textured_mesh.ply'), True).vertices,
            dtype=np.float32)
        mesh_actual = (mesh_actual - self.scale_mat[:3, 3][None]) / self.scale_mat[0, 0]
        logging.debug(mesh_actual.shape)
        mesh_actual_len = mesh_actual.shape[0]
        mesh_actual = self.split_batch(mesh_actual)
        bbox_max = np.max(mesh_actual.reshape((-1, 3)), axis=0)
        bbox_min = np.min(mesh_actual.reshape((-1, 3)), axis=0)
        logging.debug((bbox_max, bbox_min))
        logging.debug(mesh_baseline.shape)
        logging.debug(mesh_actual.shape)
        dist1, dist2, _, _ = distChamfer(torch.tensor(mesh_baseline), torch.tensor(mesh_actual))
        dist1 = dist1.numpy().reshape((-1))[:mesh_baseline_len]
        dist2 = dist2.numpy().reshape((-1))[:mesh_actual_len]
        logging.debug((dist1.shape, dist2.shape))
        logging.debug(dist1)
        logging.debug(dist2)
        accuracy = np.average(dist1)
        completeness = np.average(dist2)
        logging.debug(accuracy, completeness)
        return (accuracy + completeness) / 2


def get_blended_mvs_dataset_pair(usb_path, bmvs_model_name, processed_model_name, rewrite=False):
    """

    :param usb_path: 'E:/bmvs' or '/media/xrr/UBUNTU 22_0'
    :param bmvs_model_name: e.g. '5a7d3db14989e929563eb153'
    :param processed_model_name: e.g. 'bmvs_dog/preprocessed'
    :return:
    """
    baseline_dataset = BlendedMVSDataset(usb_path)
    if rewrite:
        baseline_dataset.set_rewrite(bmvs_model_name)
    baseline_dataset.preprocess_dataset()
    baseline_dataset.load_single_model(bmvs_model_name)

    processed_dataset = TexturedNeUSDataset('D:/城大/课程/Year 4 Sem A/CS4514/工程项目/20221224-NeuS')
    processed_dataset.process_dataset(processed_model_name)
    processed_dataset.load_model(processed_model_name)

    return baseline_dataset, processed_dataset


def get_dtu_dataset_pair(usb_path, dtu_model_name, processed_model_name, rewrite=False):
    """

    :param usb_path: 'D:/dataset/dtu', 'E:/dtu' or '/media/xrr/UBUNTU 22_0'
    :param dtu_model_name: e.g. 'scan1'
    :param processed_model_name: e.g. 'scan1'
    :return:
    """
    baseline_dataset = DTUDataset(usb_path)
    if rewrite:
        baseline_dataset.set_rewrite(dtu_model_name)
    baseline_dataset.preprocess_dataset()
    baseline_dataset.load_single_model(dtu_model_name)

    processed_dataset = TexturedNeUSDataset('D:/城大/课程/Year 4 Sem A/CS4514/工程项目/20221224-NeuS')
    processed_dataset.process_dataset(processed_model_name)
    processed_dataset.load_model(processed_model_name)

    return baseline_dataset, processed_dataset


def get_self_dataset_pair(rewrite=False):
    baseline_dataset = SelfDataset(
        'D:/城大/课程/Year 4 Sem A/CS4514/工程项目/20221224-NeuS/public_data/haibao_small/preprocessed',
        'D:/城大/课程/Year 4 Sem A/CS4514/工程项目/20221224-NeuS/exp/haibao_small/preprocessed'
    )
    baseline_dataset.preprocess_dataset(rewrite=rewrite)

    processed_dataset = TexturedNeUSDataset('D:/城大/课程/Year 4 Sem A/CS4514/工程项目/20221224-NeuS')
    processed_dataset.process_dataset('haibao_small/preprocessed')
    processed_dataset.load_model('haibao_small/preprocessed')

    return baseline_dataset, processed_dataset


def visualize_extrinsic(dataset):
    camera_previews = []
    for extrinsic in dataset.camera_extrinsics:
        preview = o3d.geometry.TriangleMesh.create_cone(radius=2, height=4)
        preview.compute_vertex_normals()
        preview = copy.deepcopy(preview).transform(extrinsic)
        camera_previews.append(preview)
    o3d.visualization.draw_geometries(camera_previews + dataset.meshes)


if __name__ == '__main__':

    metrics = Metrics('TexturedNeUSDataset_processed/scan1',
                      'DTUDataset_preprocessed/scan1')
    chamfer = metrics.ChamferL1()
    print(chamfer)
    # all_psnr = metrics.PSNR('image')`
    # print(all_psnr)
    # all_ssim = metrics.SSIM('image')
    # print(all_ssim)
    # all_lpips = metrics.LPIPS('image')
    # print(all_lpips)
