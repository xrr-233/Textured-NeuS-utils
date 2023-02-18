import os
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from open3d.visualization import gui
from PIL import Image


class Dataset:
    def __init__(self, path_root):
        self.path_root = path_root


class BlendedMVSDataset(Dataset):
    def __init__(self, path_root):
        super().__init__(path_root)
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
                self.meshes.append({
                    'name': f[:-4],
                    'geometry': mesh,
                })

        filename_root = os.path.join(filename_root, filename, filename, filename)
        self.camera_images = os.listdir(os.path.join(filename_root, 'blended_images'))
        sample_img = Image.open(os.path.join(filename_root, 'blended_images', self.camera_images[0]))
        self.W = sample_img.width
        self.H = sample_img.height

        self.camera_extrinsics = []
        self.camera_intrinsics = []
        for f in os.listdir(os.path.join(filename_root, 'cams')):
            if f == 'pair.txt':
                continue
            with open(os.path.join(filename_root, 'cams', f), 'r') as file:
                all_lines = file.readlines()
                extrinsic = np.array([
                    all_lines[1].split(' ')[:4],
                    all_lines[2].split(' ')[:4],
                    all_lines[3].split(' ')[:4],
                    all_lines[4].split(' ')[:4],
                ], dtype=np.float64)
                intrinsic = np.array([
                    all_lines[7].split(' ')[:3],
                    all_lines[8].split(' ')[:3],
                    all_lines[9].split(' ')[:3],
                ], dtype=np.float64)
                self.camera_extrinsics.append(extrinsic)
                self.camera_intrinsics.append(intrinsic)

    def visualize(self, identifier, mesh_show_back_face=True, is_filename=False):
        if is_filename:
            identifier = self.camera_images.index(f"{identifier}.jpg")
        elif identifier >= len(self.camera_images):
            print('idx exceeds max model number!')
            exit(-1)

        trajectory = []
        for i in range(len(self.camera_images)):
            pinhole_parameters = o3d.camera.PinholeCameraParameters()
            pinhole_intrinsic = o3d.camera.PinholeCameraIntrinsic()
            pinhole_intrinsic.intrinsic_matrix = self.camera_intrinsics[i]
            pinhole_parameters.intrinsic = pinhole_intrinsic
            pinhole_parameters.extrinsic = self.camera_extrinsics[i]
            trajectory.append(pinhole_parameters)
        pinhole_trajectory = o3d.camera.PinholeCameraTrajectory()
        pinhole_trajectory.parameters = trajectory

        '''renderer = o3d.visualization.Visualizer()
        renderer.create_window(width=self.W, height=self.H)
        renderer.get_render_option().mesh_show_back_face = mesh_show_back_face
        renderer.get_view_control().convert_from_pinhole_camera_parameters(pinhole_parameters, True)  # https://github.com/isl-org/Open3D/issues/1164
        for mesh in self.meshes:
            renderer.add_geometry(mesh['geometry'])
        renderer.run()
        renderer.destroy_window()
        # renderer.show(True)
        # rendered_image = renderer.render_to_image()
        # o3d.io.write_image(f'{os.getcwd()}/rendered_image.png', rendered_image)'''

        custom_draw_geometry_with_camera_trajectory = {
            'index': -1,
            'trajectory': pinhole_trajectory,
            'vis': o3d.visualization.Visualizer()
        }
        print(custom_draw_geometry_with_camera_trajectory)

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
            os.makedirs('image', exist_ok=True)
            if glb['index'] >= 0:
                print(f"Capture image {self.camera_images[glb['index']][:-4]}.png")
                vis.capture_screen_image(f"image/{self.camera_images[glb['index']][:-4]}.png", False)
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
            vis.add_geometry(mesh['geometry'])
        vis.get_render_option().mesh_show_back_face = mesh_show_back_face
        vis.register_animation_callback(move_forward)
        vis.run()
        vis.destroy_window()

        sample_img = Image.open(os.path.join(f"image/00000000.png"))
        print(sample_img.width, sample_img.height)


class TexturedNeUSDataset(Dataset):
    pass


if __name__ == '__main__':
    dataset = BlendedMVSDataset('E:')
    # dataset = BlendedMVSDataset('/media/xrr/UBUNTU 22_0')
    dataset.load_single_model(1)  # 以大鼎为例
    dataset.visualize('00000003', is_filename=True)
