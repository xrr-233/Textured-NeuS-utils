import os
import numpy as np
import open3d as o3d
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
        elif identifier >= self.__len__():
            print('idx exceeds max model number!')
            exit(-1)
        pinhole_parameters = o3d.camera.PinholeCameraParameters()
        pinhole_intrinsic = o3d.camera.PinholeCameraIntrinsic()
        pinhole_intrinsic.intrinsic_matrix = self.camera_intrinsics[identifier]
        pinhole_parameters.intrinsic = pinhole_intrinsic
        pinhole_parameters.extrinsic = self.camera_extrinsics[identifier]
        renderer = o3d.visualization.Visualizer()
        renderer.create_window(width=self.W, height=self.H)
        renderer.get_render_option().mesh_show_back_face = mesh_show_back_face
        print(renderer.get_view_control())
        renderer.get_view_control().convert_from_pinhole_camera_parameters(pinhole_parameters, True)  # https://github.com/isl-org/Open3D/issues/1164
        for mesh in self.meshes:
            renderer.add_geometry(mesh['geometry'])
        renderer.run()
        renderer.destroy_window()
        # renderer.show(True)
        # rendered_image = renderer.render_to_image()
        # o3d.io.write_image(f'{os.getcwd()}/rendered_image.png', rendered_image)
        # o3d.visualization.draw_geometries(self.meshes, mesh_show_back_face=mesh_show_back_face)


class TexturedNeUSDataset(Dataset):
    pass


if __name__ == '__main__':
    dataset = BlendedMVSDataset('E:')
    # dataset = BlendedMVSDataset('/media/xrr/UBUNTU 22_0')
    dataset.load_single_model(1)  # 以大鼎为例
    dataset.visualize('00000003', is_filename=True)
