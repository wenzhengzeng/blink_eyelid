import os
import numpy as np

import cv2

import torch
import torch.utils.data as data
from utils.osutils import *
from utils.imutils import *
from utils.transforms import *
from .exam import rechecktest
import torchvision.transforms as transforms



class HUST_LEBW(data.Dataset):
    def __init__(self, cfg, train=True):
        self.img_folder = cfg.root_path
        self.is_train = train
        self.inp_res = cfg.data_shape
        self.out_res = cfg.output_shape
        self.pixel_means = cfg.pixel_means
        self.num_class = cfg.num_class
        self.cfg = cfg
        self.bbox_extend_factor = cfg.bbox_extend_factor
        if train:
            self.scale_factor = cfg.scale_factor
            self.rot_factor = cfg.rot_factor
            self.symmetry = cfg.symmetry

        self.eye = cfg.eye

        self.blink_gt_path = os.path.join(self.img_folder, 'check', f'gt_blink_{self.eye}.txt')
        self.non_blink_gt_path = os.path.join(self.img_folder, 'check', f'gt_non_blink_{self.eye}.txt')

        with open(self.blink_gt_path, "r") as blink_file:
            blink = blink_file.readlines()

        with open(self.non_blink_gt_path, "r") as non_blink_file:
            non_blink = non_blink_file.readlines()
        self.anno = blink + non_blink

        


    def augmentationCropImage(self, img, joints=None):
        height, width = self.inp_res[0], self.inp_res[1]
        
        img = cv2.resize(img, (width, height))

        return img


    def data_augmentation(self, img, leftmap, affrat, angle):

        leftmap = cv2.resize(leftmap, (192, 256))

        img = img.astype(np.float32) / 255

        left_eye = np.mean(np.where(leftmap == np.max(leftmap)), 1)

        return img, left_eye

    def __getitem__(self, index):
        image_name = self.anno[index].strip(' \r\n')
        image_path = image_name.split(' ')
        images = []
        eye_poses = []
        blink_label = torch.tensor(int(image_path[0]))
        pos_path=image_path[1].split('/')

        with open(self.img_folder + '/check/' + pos_path[1] + '/' + pos_path[2] + '/10/eye_pos_relative.txt',"r") as pos_file:
            pos = pos_file.readlines()

        for i in range(1, 11):
            img_path = os.path.join(self.img_folder, image_path[i])
            img_path = img_path.strip('.bmp')
            img_path = img_path+'face.bmp'

            image = cv2.imread(img_path)

            pos_cur = pos[i - 1].strip(' \n')
            pos_cur = pos_cur.split(' ')
            if self.eye == 'right':
                eye_pos = torch.tensor([int(float(pos_cur[3]) / image.shape[1] * 192), int(float(pos_cur[4]) / image.shape[0] * 256)])
            else:
                 eye_pos = torch.tensor([int(float(pos_cur[1]) / image.shape[1] * 192), int(float(pos_cur[2]) / image.shape[0] * 256)])
            eye_pos[0] = min(eye_pos[0], 191-50)
            eye_pos[0] = max(eye_pos[0], 50)
            eye_pos[1] = min(eye_pos[1], 255 - 50)
            eye_pos[1] = max(eye_pos[1], 50)
            eye_poses.append(eye_pos)

            img = self.augmentationCropImage(image)

            img = img[...,::-1].copy()
            img = img.astype(np.float32) / 255
            img = im_to_torch(img)
            img = color_normalize(img, self.pixel_means)

            images.append(img)


        imgs = torch.stack(images)
        eyeposes = torch.stack(eye_poses)
        return imgs, eyeposes, blink_label

    def __len__(self):
        return len(self.anno)


