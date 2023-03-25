import pickle
import matplotlib.pyplot as plt
import numpy as np

def load_npy(path):
    loadData = np.load(path)

    print("----type----")
    print(type(loadData))
    print("----shape----")
    print(loadData.shape)
    print("----data----")
    print(loadData)

def load_npz(path):
    """
    打印在npz格式的数据中存在的所有类dict属性

    :param path: # e.g. npz_path = 'data/demo/Matterport3D_processed/17DRP5sb8fy/pointcloud.npz'
    :return:
    """
    print("Load an npz file and print it")
    npz_point_cloud = np.load(path)
    for key in npz_point_cloud:
        print(key)
        print(npz_point_cloud[key].shape)
        print(npz_point_cloud[key])

def load_pickle(path):
    """
    打印pkl格式文件（类似一个table）
    :param path: # e.g. pkl_path = './out/demo_matterport/generation/time_generation_full.pkl'
    :return:
    """
    f = open(path,'rb')
    data = pickle.load(f)
    print(data)

def lr():
    learning_rate = 5e-4
    learning_rate_alpha = 0.05
    warm_up_end = 5000
    end_iter = 300000
    res = []
    for iter_step in range(end_iter):
        if iter_step < warm_up_end:
            learning_factor = iter_step / warm_up_end
        else:
            alpha = learning_rate_alpha
            progress = (iter_step - warm_up_end) / (end_iter - warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        res.append(learning_rate * learning_factor)
    plt.plot(res)
    plt.savefig('x.png')

if (__name__=="__main__"):
    npy_path = 'poses.npy'
    load_npy(npy_path)
