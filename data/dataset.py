from torch.utils.data import Dataset
import torchvision.transforms as T
import glob
import random
import config as cfg
import os.path as osp
from imutils import paths
import cv2
import numpy as np
from PIL import Image

def read_image(img_path, part_size=0, rand_ch=None):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    if img_path == 'no':
        h = part_size[1]
        w = part_size[0]
        img = np.zeros((h, w, 3), np.uint8)
        return Image.fromarray(img, mode='RGB')
    else:
        got_img = False
        if not osp.exists(img_path):
            raise IOError("{} does not exist".format(img_path))
        while not got_img:
            try:
                img = Image.open(img_path).convert('RGB')

                if rand_ch and random.random() > rand_ch:
                    img = Image.fromarray(np.asarray(img)[:, :, ::-1])

                got_img = True
            except IOError:
                print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
                pass
        return img


class ImageDataSet(Dataset):
    def __init__(self, root, classes_dict, transform=None, target_transform=None, is_train=False):
        super(ImageDataSet, self).__init__()

        self.class_dict = classes_dict

        # class dict
        self.image_paths = list(paths.list_images(root))

        random.shuffle(self.image_paths)

        # get label
        self.label = [self.class_dict[p.split('/')[-2]] for p in self.image_paths]

        self.transform = transform
        self.label_transform = target_transform
        self.is_train = is_train

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_p = self.image_paths[index]
        label = self.label[index]

        if self.is_train:
            img = read_image(img_p, rand_ch=0.5)
        else:
            img = read_image(img_p)

        if self.transform is not None:
            img = self.transform(img)
 
        return img, label, img_p

