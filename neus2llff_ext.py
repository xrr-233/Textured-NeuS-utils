import os
import shutil
import trimesh
import numpy as np
import open3d as o3d
from PIL import Image

from colmap_utils import colmap_read_model
from utils import load_K_Rt_from_P


def load_colmap_data(realdir):
    camerasfile = os.path.join(realdir, 'sparse/0/cameras.bin')
    camdata = colmap_read_model.read_cameras_binary(camerasfile)

    # cam = camdata[camdata.keys()[0]]
    list_of_keys = list(camdata.keys())
    cam = camdata[list_of_keys[0]]
    print('Cameras', len(cam))

    h, w, f = cam.height, cam.width, cam.params[0]
    # w, h, f = factor * w, factor * h, factor * f
    hwf = np.array([h, w, f]).reshape([3, 1])

    imagesfile = os.path.join(realdir, 'sparse/0/images.bin')
    imdata = colmap_read_model.read_images_binary(imagesfile)

    w2c_mats = []
    bottom = np.array([0, 0, 0, 1.]).reshape([1, 4])

    names = [imdata[k].name for k in imdata]
    print('Images #', len(names))
    perm = np.argsort(names)
    for k in imdata:
        im = imdata[k]
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3, 1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)

    w2c_mats = np.stack(w2c_mats, 0)
    c2w_mats = np.linalg.inv(w2c_mats)

    poses = c2w_mats[:, :3, :4].transpose([1, 2, 0])
    poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1, 1, poses.shape[-1]])], 1)

    points3dfile = os.path.join(realdir, 'sparse/0/points3D.bin')
    pts3d = colmap_read_model.read_points3d_binary(points3dfile)

    # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]],
                           1)

    return poses, pts3d, perm


def save_poses(poses, pts3d, perm):
    pts_arr = []
    vis_arr = []
    for k in pts3d:
        pts_arr.append(pts3d[k].xyz)
        cams = [0] * poses.shape[-1]
        for ind in pts3d[k].image_ids:
            if len(cams) < ind - 1:
                print('ERROR: the correct camera poses for current points cannot be accessed')
                return
            cams[ind - 1] = 1
        vis_arr.append(cams)

    pts_arr = np.array(pts_arr)
    vis_arr = np.array(vis_arr)
    print('Points', pts_arr.shape, 'Visibility', vis_arr.shape)

    poses = np.moveaxis(poses, -1, 0)
    poses = poses[perm]
    return poses


def gen_cameras(poses_ply, work_dir):
    poses_hwf = poses_ply  # n_images, 3, 5
    poses_raw = poses_hwf[:, :, :4]
    hwf = poses_hwf[:, :, 4]
    pose = np.diag([1.0, 1.0, 1.0, 1.0])
    pose[:3, :4] = poses_raw[0]

    cam_dict = dict()
    n_images = len(poses_raw)

    # Convert space
    convert_mat = np.zeros([4, 4], dtype=np.float32)
    convert_mat[0, 1] = 1.0
    convert_mat[1, 0] = 1.0
    convert_mat[2, 2] = -1.0
    convert_mat[3, 3] = 1.0

    for i in range(n_images):
        pose = np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float32)
        pose[:3, :4] = poses_raw[i]
        pose = pose @ convert_mat
        h, w, f = hwf[i, 0], hwf[i, 1], hwf[i, 2]
        intrinsic = np.diag([f, f, 1.0, 1.0]).astype(np.float32)
        intrinsic[0, 2] = (w - 1) * 0.5
        intrinsic[1, 2] = (h - 1) * 0.5
        w2c = np.linalg.inv(pose)
        world_mat = intrinsic @ w2c
        world_mat = world_mat.astype(np.float32)
        cam_dict['camera_mat_{}'.format(i)] = intrinsic
        cam_dict['camera_mat_inv_{}'.format(i)] = np.linalg.inv(intrinsic)
        cam_dict['world_mat_{}'.format(i)] = world_mat
        cam_dict['world_mat_inv_{}'.format(i)] = np.linalg.inv(world_mat)

    pcd = trimesh.load(os.path.join(work_dir, 'textured_mesh.ply'))
    vertices = pcd.vertices
    bbox_max = np.max(vertices, axis=0)
    bbox_min = np.min(vertices, axis=0)
    center = (bbox_max + bbox_min) * 0.5
    radius = np.linalg.norm(vertices - center, ord=2, axis=-1).max()
    scale_mat = np.diag([radius, radius, radius, 1.0]).astype(np.float32)
    scale_mat[:3, 3] = center

    for i in range(n_images):
        cam_dict['scale_mat_{}'.format(i)] = scale_mat
        cam_dict['scale_mat_inv_{}'.format(i)] = np.linalg.inv(scale_mat)

    np.savez(os.path.join(work_dir, 'cameras_sphere.npz'), **cam_dict)
    print('Preprocess done!')

class Renderer():

    def load_camera_parameters(self, src_root):
        """
        指定某一个模型，导入并储存其相机参数，用以渲染和生成camera_sphere.npz
        :param src_root: 模型主文件夹路径
        :return:
        """
        self.meshes = [o3d.io.read_triangle_mesh(os.path.join(src_root, 'textured_mesh.ply'), True)]
        self.meshes[0].compute_vertex_normals()  # allow light effect

        camera_images = os.listdir(os.path.join(src_root, 'image'))
        self.n_images = len(camera_images)
        sample_img = Image.open(os.path.join(src_root, 'image', camera_images[0]))
        self.W = sample_img.width
        self.H = sample_img.height

        camera_dict = np.load(os.path.join(src_root, 'cameras_sphere.npz'))
        world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.camera_intrinsics = []
        self.camera_extrinsics = []
        for scale_mat, world_mat in zip(scale_mats_np, world_mats_np):
            P = world_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.camera_intrinsics.append(intrinsics[:3, :3])
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
            pinhole_parameters.extrinsic = np.linalg.inv(self.camera_extrinsics[i])
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
                print(glb['trajectory'].parameters[glb['index']].extrinsic)
                ctr.convert_from_pinhole_camera_parameters(
                    glb['trajectory'].parameters[glb['index']], allow_arbitrary=True)
            else:
                self.custom_draw_geometry_with_camera_trajectory['vis'].close()

        vis = self.custom_draw_geometry_with_camera_trajectory['vis']
        vis.create_window(width=self.W, height=self.H)
        for mesh in self.meshes:
            vis.add_geometry(mesh)
        vis.get_render_option().mesh_show_back_face = mesh_show_back_face
        vis.register_animation_callback(move_forward)
        vis.run()
        vis.destroy_window()


if __name__ == "__main__":
    os.makedirs('SPSR_processed')
    all_dtu = os.listdir(
        os.path.join('neus2llff_preprocessed', 'TexturedNeUSDataset_processed', 'DTUDataset_preprocessed'))
    all_cases_parent = []
    all_cases = []
    for case in all_dtu:
        all_cases_parent.append('DTUDataset_preprocessed')
        all_cases.append(case)

    for i in range(len(all_cases)):
        src = os.path.join('neus2llff_preprocessed', 'TexturedNeUSDataset_processed', all_cases_parent[i], all_cases[i])
        path_root = os.path.join('SPSR_processed', all_cases_parent[i], all_cases[i])
        os.makedirs(path_root)
        shutil.copytree(os.path.join(src, 'images'), os.path.join(path_root, 'image'))
        shutil.copy(os.path.join(src, 'dense', 'meshed-poisson.ply'), os.path.join(path_root, 'textured_mesh.ply'))

        poses, pts3d, perm = load_colmap_data(src)
        poses_npy = save_poses(poses, pts3d, perm)
        gen_cameras(poses_npy, path_root)

        os.makedirs(os.path.join(path_root, 'image_mesh'))
        renderer = Renderer()
        renderer.load_camera_parameters(path_root)
        renderer.generate_baseline_rendered_mesh(save_path=path_root)
