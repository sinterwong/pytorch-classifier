from torch.utils.data import Dataset, Sampler
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import glob
import random
import config as cfg
import os.path as osp
from imutils import paths
import cv2
import numpy as np
from PIL import Image
import collections

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


class ImageDataSet2(Dataset):
    def __init__(self, root, classes_dict, transform=None, data_aug=None, is_train=False):
        super(ImageDataSet2, self).__init__()

        self.class_dict = classes_dict

        # class dict
        self.image_paths = np.array(list(paths.list_images(root)))
        random.shuffle(self.image_paths)

        # get label
        self.label = [self.class_dict[p.split('/')[-2]] for p in self.image_paths]

        if transform is not None:
            self.transform = transform
        else:
            self.transform = T.ToTensor()
        self.data_aug = data_aug
        self.is_train = is_train
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_p = self.image_paths[index]
        label = self.label[index]

        if self.is_train:
            img = cv2.imread(img_p)
            if self.data_aug:
                img = self.data_aug(images=np.expand_dims(img, axis=0))
                img = img.squeeze(0)
        else:
            img = cv2.imread(img_p)

        img = Image.fromarray(img[:, :, ::-1])  # bgr -> rgb for PIL
        data = self.transform(img)
 
        return data, label, img_p


class ImageDatasetWrapper(Dataset):
    # Additinoal attributes
    # - indices
    # - classwise_indices
    # - num_classes
    # - get_class

    def __init__(self, dataset, indices=None):
        self.base_dataset = dataset
        if indices is None:
            self.indices = list(range(len(dataset)))
        else:
            self.indices = indices

        # label 对应的id, 用于 triplet loss 或 自蒸馏等
        self.classwise_indices = {}
        # id 对应的 label
        self.idx2lname = {}
        for i, v in enumerate(self.base_dataset.label):
            self.classwise_indices.setdefault(v, [])
            self.classwise_indices[v].append(i)
            self.idx2lname[i] = v

        self.num_classes = max(self.classwise_indices.keys())+1

    def __getitem__(self, i):
        return self.base_dataset[self.indices[i]]

    def __len__(self):
        return len(self.indices)

    def get_class(self, idx):
        return self.idx2lname[idx]


class PairBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_iterations=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_iterations = num_iterations

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        for k in range(len(self)):
            if self.num_iterations is None:
                offset = k*self.batch_size
                batch_indices = indices[offset:offset+self.batch_size]
            else:
                batch_indices = random.sample(range(len(self.dataset)),
                                              self.batch_size)

            pair_indices = []
            for idx in batch_indices:
                y = self.dataset.get_class(idx)
                pair_indices.append(random.choice(self.dataset.classwise_indices[y]))

            yield batch_indices + pair_indices

    def __len__(self):
        if self.num_iterations is None:
            return (len(self.dataset)+self.batch_size-1) // self.batch_size
        else:
            return self.num_iterations

