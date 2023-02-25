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
        pass

    def generate_baseline_rendered_mesh(self, mesh_show_back_face=True, save_path='.'):
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

        custom_draw_geometry_with_camera_trajectory = {
            'index': -1,
            'trajectory': pinhole_trajectory,
            'vis': o3d.visualization.Visualizer()
        }

        def move_forward(vis):
            # This function is called within the o3d.visualization.Visualizer::run() loop
            # The run loop calls the function, then re-render
            # So the sequence in this function is to:
            # 1. Capture frame
            # 2. index++, check ending criteria
            # 3. Set camera
            # 4. (Re-render)
            ctr = vis.get_view_control()
            glb = custom_draw_geometry_with_camera_trajectory
            if glb['index'] >= 0:
                print(os.path.join(save_path, 'image_mesh', '{:0>3d}.png'.format(glb['index'])))
                vis.capture_screen_image(os.path.join(save_path, 'image_mesh', '{:0>3d}.png'.format(glb['index'])),
                                         False)
            glb['index'] = glb['index'] + 1
            if glb['index'] < len(glb['trajectory'].parameters):
                ctr.convert_from_pinhole_camera_parameters(
                    glb['trajectory'].parameters[glb['index']], allow_arbitrary=True)
            else:
                custom_draw_geometry_with_camera_trajectory['vis']. \
                    register_animation_callback(None)
            return False

        vis = custom_draw_geometry_with_camera_trajectory['vis']
        print(self.W, self.H)
        vis.create_window(width=self.W, height=self.H)
        for mesh in self.meshes:
            vis.add_geometry(mesh)
        vis.get_render_option().mesh_show_back_face = mesh_show_back_face
        vis.register_animation_callback(move_forward)
        vis.run()
        vis.destroy_window()


class BlendedMVSDataset(Dataset):
    def __init__(self, path_root):
        super().__init__()
        self.image_dirs = []
        for filename in os.listdir(path_root):
            if filename.startswith('dataset_full_res_'):
                self.image_dirs.append(os.path.join(path_root, filename))
        self.textured_mesh_dir = os.path.join(path_root, 'dataset_textured_meshes')

        all_models = []
        self.all_models_root = []
        self.all_models = []
        for image_dir in self.image_dirs:
            all_model = os.listdir(image_dir)
            self.all_models_root.extend(image_dir for _ in range(len(all_model)))
            all_models.extend(all_model)
        for filename in os.listdir(self.textured_mesh_dir):
            if filename in all_models:
                self.all_models.append(filename)

    def __len__(self):
        return len(self.all_models)

    def load_single_model(self, identifier, is_filename=False):
        # device = o3d.core.Device("CPU:0")
        filename_root = None
        filename = None
        if is_filename:
            filename_root = self.all_models_root[self.all_models.index(identifier)]
            filename = identifier
        else:
            if identifier < self.__len__():
                filename_root = self.all_models_root[identifier]
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

        filename_root = os.path.join(filename_root, filename, filename, filename)
        self.camera_images = os.listdir(os.path.join(filename_root, 'blended_images'))
        self.n_images = len(self.camera_images)
        sample_img = Image.open(os.path.join(filename_root, 'blended_images', self.camera_images[0]))
        self.W = sample_img.width
        self.H = sample_img.height

    def preprocess_dataset(self):
        os.makedirs('BlendedMVS_preprocessed', exist_ok=True)

        for i in tqdm(range(self.__len__())):
            filename_root = self.all_models_root[i]
            filename = self.all_models[i]
            os.makedirs(f'BlendedMVS_preprocessed/{filename}', exist_ok=True)
            os.makedirs(f'BlendedMVS_preprocessed/{filename}/preprocessed', exist_ok=True)
            os.makedirs(f'BlendedMVS_preprocessed/{filename}/preprocessed/image', exist_ok=True)
            os.makedirs(f'BlendedMVS_preprocessed/{filename}/preprocessed/image_mesh', exist_ok=True)
            os.makedirs(f'BlendedMVS_preprocessed/{filename}/preprocessed/mask', exist_ok=True)

            src_root = os.path.join(filename_root, filename, filename, filename, 'blended_images')
            dst_root = f'BlendedMVS_preprocessed/{filename}/preprocessed'

            for index, file in enumerate(os.listdir(src_root)):
                img = cv2.imread(os.path.join(src_root, file))
                cv2.imwrite(os.path.join(dst_root, 'image', '{:0>3d}.png'.format(index)), img)
                cv2.imwrite(os.path.join(dst_root, 'mask', '{:0>3d}.png'.format(index)), np.ones_like(img) * 255)

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
            self.load_single_model(filename, is_filename=True)
            # visualizer = CameraPoseVisualizer([-2, 2], [-2, 2], [-2, 2])
            # for i in range(self.n_images):
            #     visualizer.extrinsic2pyramid(self.camera_extrinsics[i], focal_len_scaled=0.25, aspect_ratio=0.3)
            # visualizer.show()
            # self.generate_baseline_rendered_mesh(save_path=dst_root)

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

            for i in range(self.n_images):
                cam_dict['scale_mat_{}'.format(i)] = scale_mat
                cam_dict['scale_mat_inv_{}'.format(i)] = np.linalg.inv(scale_mat)

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

            np.savez(os.path.join(dst_root, 'cameras_sphere.npz'), **cam_dict)


class DTUOrSelfDataset(Dataset):
    def __init__(self, path_public, path_exp):
        super().__init__()
        self.path_root = 'DTUOrSelfDataset_preprocessed'
        self.path_public = path_public
        self.path_exp = path_exp

    def preprocess_dataset(self, rewrite=False):
        os.makedirs(self.path_root, exist_ok=True)
        os.makedirs(f'{self.path_root}/image_mesh', exist_ok=True)

        if os.path.exists(os.path.join(self.path_root, 'image')) and rewrite:
            shutil.rmtree(os.path.join(self.path_root, 'image'))
        if os.path.exists(os.path.join(self.path_root, 'mask')) and rewrite:
            shutil.rmtree(os.path.join(self.path_root, 'mask'))
        if os.path.exists(os.path.join(self.path_root, 'cameras_sphere.npz')) and rewrite:
            os.remove(os.path.join(self.path_root, 'cameras_sphere.npz'))
        if os.path.exists(os.path.join(self.path_root, 'textured_mesh.ply')) and rewrite:
            os.remove(os.path.join(self.path_root, 'textured_mesh.ply'))

        if not os.path.exists(os.path.join(self.path_root, 'image')):
            shutil.copytree(os.path.join(self.path_public, 'image'),
                            os.path.join(self.path_root, 'image'))
        if not os.path.exists(os.path.join(self.path_root, 'mask')):
            shutil.copytree(os.path.join(self.path_public, 'mask'),
                            os.path.join(self.path_root, 'mask'))
        if not os.path.exists(os.path.join(self.path_root, 'cameras_sphere.npz')):
            shutil.copy(os.path.join(self.path_public, 'cameras_sphere.npz'),
                        os.path.join(self.path_root, 'cameras_sphere.npz'))
        if not os.path.exists(os.path.join(self.path_root, 'textured_mesh.ply')):
            shutil.copy(os.path.join(self.path_exp, 'womask_sphere', 'meshes', 'vertex_color.ply'),
                        os.path.join(self.path_root, 'textured_mesh.ply'))

        self.camera_images = os.listdir(os.path.join(self.path_root, 'image'))
        self.n_images = len(self.camera_images)
        sample_img = Image.open(os.path.join(self.path_root, 'image', self.camera_images[0]))
        self.W = sample_img.width
        self.H = sample_img.height

        camera_dict = np.load(os.path.join(self.path_root, 'cameras_sphere.npz'))
        world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        self.camera_intrinsics = []
        self.camera_extrinsics = []
        for scale_mat, world_mat in zip(scale_mats_np, world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.camera_intrinsics.append(intrinsics[:3, :3])
            self.camera_extrinsics.append(pose)
        self.meshes = [o3d.io.read_triangle_mesh(os.path.join(self.path_root, 'textured_mesh.ply'), True)]


class TexturedNeUSDataset(Dataset):
    def __init__(self, path_baseline, path_exp):
        super().__init__()
        self.path_root = 'TexturedNeUSDataset_processed'
        self.path_baseline = path_baseline
        self.path_exp = path_exp

    def process_dataset(self, counterpart, rewrite=False):
        os.makedirs(self.path_root, exist_ok=True)
        os.makedirs(f'{self.path_root}/image_mesh', exist_ok=True)

        if os.path.exists(os.path.join(self.path_root, 'image')) and rewrite:
            shutil.rmtree(os.path.join(self.path_root, 'image'))
        if os.path.exists(os.path.join(self.path_root, 'cameras_sphere.npz')) and rewrite:
            os.remove(os.path.join(self.path_root, 'cameras_sphere.npz'))
        if os.path.exists(os.path.join(self.path_root, 'textured_mesh.ply')) and rewrite:
            os.remove(os.path.join(self.path_root, 'textured_mesh.ply'))
        if os.path.exists(os.path.join(self.path_root, 'ckpt.pth')) and rewrite:
            os.remove(os.path.join(self.path_root, 'ckpt.pth'))

        if not os.path.exists(os.path.join(self.path_root, 'ckpt.pth')):
            model_list_raw = os.listdir(os.path.join(self.path_exp, 'womask_sphere', 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth':
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]
            shutil.copy(os.path.join(self.path_exp, 'womask_sphere', 'checkpoints', latest_model_name),
                        os.path.join(self.path_root, 'ckpt.pth'))
        if not os.path.exists(os.path.join(self.path_root, 'cameras_sphere.npz')):
            shutil.copy(os.path.join(self.path_baseline, 'cameras_sphere.npz'),
                        os.path.join(self.path_root, 'cameras_sphere.npz'))
        if not os.path.exists(os.path.join(self.path_root, 'image')):
            os.makedirs(os.path.join(self.path_root, 'image'), exist_ok=True)
            self.runner = Runner('', counterpart.n_images, counterpart.W, counterpart.H)
            self.runner.load_camera_params(os.path.join(self.path_root, 'cameras_sphere.npz'))
            self.runner.load_ckpt(os.path.join(self.path_root, 'ckpt.pth'))
            self.runner.validate_image(save_path=self.path_root)
        if not os.path.exists(os.path.join(self.path_root, 'textured_mesh.ply')):
            shutil.copy(os.path.join(self.path_exp, 'womask_sphere', 'meshes', 'vertex_color.ply'),
                        os.path.join(self.path_root, 'textured_mesh.ply'))

        self.camera_images = os.listdir(os.path.join(self.path_root, 'image'))
        self.n_images = len(self.camera_images)
        sample_img = Image.open(os.path.join(self.path_root, 'image', self.camera_images[0]))
        self.W = sample_img.width
        self.H = sample_img.height

        camera_dict = np.load(os.path.join(self.path_root, 'cameras_sphere.npz'))
        world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        self.camera_intrinsics = []
        self.camera_extrinsics = []
        for scale_mat, world_mat in zip(scale_mats_np, world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.camera_intrinsics.append(intrinsics[:3, :3])
            self.camera_extrinsics.append(pose)
        self.meshes = [o3d.io.read_triangle_mesh(os.path.join(self.path_root, 'textured_mesh.ply'), True)]


class Metrics:
    def __init__(self, baseline_path, actual_path):
        self.baseline_path = baseline_path
        self.actual_path = actual_path

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


if __name__ == '__main__':
    # dataset = BlendedMVSDataset('E:')
    # dataset = BlendedMVSDataset('/media/xrr/UBUNTU 22_0')
    baseline_dataset = DTUOrSelfDataset(
        'D:/城大/课程/Year 4 Sem A/CS4514/工程项目/20221224-NeuS/public_data/haibao/preprocessed',
        'D:/城大/课程/Year 4 Sem A/CS4514/工程项目/20221224-NeuS/exp/haibao/preprocessed'
    )
    baseline_dataset.preprocess_dataset()
    # baseline_dataset.generate_baseline_rendered_mesh(save_path=baseline_dataset.path_root)
    processed_dataset = TexturedNeUSDataset(
        baseline_dataset.path_root,
        'D:/城大/课程/Year 4 Sem A/CS4514/工程项目/20221224-NeuS/exp/haibao/preprocessed'
    )
    processed_dataset.process_dataset(baseline_dataset, rewrite=True)
    # processed_dataset.generate_baseline_rendered_mesh(save_path=processed_dataset.path_root)

    # metrics = Metrics(baseline_dataset.path_root, processed_dataset.path_root)
    # all_psnr = metrics.PSNR('image')`
    # print(all_psnr)
    # all_ssim = metrics.SSIM('image')
    # print(all_ssim)
    # all_lpips = metrics.LPIPS('image')
    # print(all_lpips)
