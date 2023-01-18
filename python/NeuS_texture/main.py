import os
import logging
import torch
import numpy as np
import trimesh

from pyhocon import ConfigFactory
from NeuS_texture.models.fields import NeRF, SDFNetwork, SingleVarianceNetwork, RenderingNetwork


class Runner():
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

    def load_ckpt(self):
        latest_model_name = None
        model_list_raw = os.listdir('data/ckpts')
        model_list = []
        for model_name in model_list_raw:
            if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= 999999:
                model_list.append(model_name)
        model_list.sort()

        if(len(model_list) > 0):
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

            model = trimesh.load(os.path.join('data/meshes', latest_model_name))
            self.vertices = model.vertices
            self.vertices = (self.vertices - self.scale_mats_np[0][:3, 3][None]) / self.scale_mats_np[0][0, 0]
        else:
            logging.error('No 3d model found')
            exit(-1)

if(__name__=="__main__"):
    logging.basicConfig(level=logging.DEBUG)

    runner = Runner('haibao')

    runner.load_camera_params()
    runner.load_mesh()

    print(np.min(runner.vertices, axis=0))
    print(np.max(runner.vertices, axis=0))