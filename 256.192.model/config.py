import os
import os.path
import sys
import numpy as np


def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)


class Config:
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    this_dir_name = cur_dir.split('/')[-1]
    root_dir = os.path.join(cur_dir, '..')
    data_dir = "/data/data1/zengwenzheng/HUST_LEBW/TIFS2023_release/HUST_LEBW/"  
    model = 'CPN18'

    lr = 5e-4
    lr_gamma = 0.8
    lr_dec_epoch = list(range(6, 4000, 6))
    num_clusters = 4
    batch_size = 1
    weight_decay = 1e-5
    time_size = 10

    num_class = 2
    root_path = os.path.join(data_dir, 'test')
    symmetry = [(0, 1)]
    bbox_extend_factor = (0.1, 0.15)  # x, y:

    # data augmentation setting
    scale_factor = (0.95, 1.05)
    rot_factor = 40

    pixel_means = np.array([78.2947, 62.2556, 50.3173])  # RGB

    data_shape = (256, 192)
    output_shape = (64, 48)



cfg = Config()
add_pypath(cfg.root_dir)

