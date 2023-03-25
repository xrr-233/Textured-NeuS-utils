import os
import shutil
import numpy as np
import cv2
from PIL import Image


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
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


def resize(path):
    os.makedirs('temp', exist_ok=True)
    all_imgs = os.listdir(path)
    for img in all_imgs:
        im = cv2.imread(os.path.join(path, img))
        im = cv2.resize(im, (0, 0), fx=0.5, fy=0.5)
        cv2.imwrite(os.path.join('temp', img), im)


def integrate_imgs():
    imgs_path = 'NeuS_texture/data/imgs_original'
    masks_path = './NeuS_texture/data/masks'

    imgs_dir = os.listdir(imgs_path)
    masks_dir = os.listdir(masks_path)

    res_path = './NeuS_texture/data/imgs_masks'
    os.makedirs(res_path, exist_ok=True)

    for i in range(len(imgs_dir)):
        img = Image.open(os.path.join(imgs_path, imgs_dir[i]))
        mask = Image.open(os.path.join(masks_path, masks_dir[i]))

        img = np.array(img)
        mask = np.array(mask)

        res = img & mask
        res = Image.fromarray(res)
        res.save(os.path.join(res_path, imgs_dir[i]))

def sample_bmvs():
    """
    Used to find small object scenes in BlendedMVS.
    """
    root_path = "D:\CityU\Courses\Year4SemA\CS4514\Projects\BlendedMVS"
    os.makedirs('sample_bmvs', exist_ok=True)

    for case in os.listdir(root_path):
        images = os.listdir(os.path.join(root_path, case, 'blended_images'))
        shutil.copy(os.path.join(root_path, case, 'blended_images', images[1]), os.path.join('sample_bmvs', f'{case}.png'))