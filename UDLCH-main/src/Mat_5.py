import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import h5py
import scipy.io as sio
import pdb

class Sampler():
    def __init__(self, root, paths):
        self.root = root
        if isinstance(paths, np.ndarray):
            if len(paths.shape) == 1 or paths.shape[0] == 1 or paths.shape[1] == 1:
                paths = paths.reshape([-1]).tolist()
        self.paths = paths

    def __getitem__(self, item):
        path = self.paths[item]

        if isinstance(path, Image.Image):
            return path

        if isinstance(path, np.ndarray):
            if len(path.shape) >= 2:
                return Image.fromarray(path, mode='RGB')
            else:
                path = path[0]

        return Image.open(os.path.join(self.root, path))

    def __len__(self):
        return len(self.paths)


def text_transform(text):
    return text


class CMDataset(Dataset):
    def __init__(self, data_name, batch_size=64, category_split_ratio=(23, 1)):
        self.data_name = data_name
        self.batch_size = batch_size
        self.category_split_ratio = category_split_ratio
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        trans = [
            transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        ]
        self.trans = trans

        self.open_data()

    def get_all_data(self):

        all_images = []
        all_texts = []
        all_labels = self.labels


        if isinstance(self.imgs, Sampler):
            all_images = [self.imgs[i] for i in range(len(self.imgs))]
        else:
            all_images = self.imgs


        all_texts = self.texts

        return all_images, all_texts, all_labels

    def open_data(self):

        if self.data_name.lower() == 'mirflickr25k':
            data = MIRFlickr25K()
        elif self.data_name.lower() == 'mirflickr25k_fea':
            data = MIRFlickr25K_fea()
        elif self.data_name.lower() == 'nuswide_fea':
            data = NUSWIDE_fea()
        elif self.data_name.lower() == 'nuswide':
            data = NUSWIDE()
        elif self.data_name.lower() == 'mscoco_fea':
            data = MSCOCO_fea()

        if len(data) == 3:
            (self.imgs, self.texts, self.labels) = data
        else:
            (self.imgs, self.texts, self.labels, root) = data
            self.imgs = Sampler(root, self.imgs)
            if not isinstance(self.imgs, Sampler):
                self.imgs = self.imgs


        self.split_by_categories()


        self.create_retrieval_and_query_sets()

    def split_by_categories(self):

        num_samples, num_classes = self.labels.shape

        category_counts = np.sum(self.labels, axis=0)
        print(f'category_counts={category_counts}')
        print(f'len={len(category_counts)}')


        sorted_categories = sorted(range(num_classes), key=lambda x: category_counts[x], reverse=False)

        print(f'sorted_categories={sorted_categories}')


        visible_categories_count = self.category_split_ratio[0]
        invisible_categories_count = self.category_split_ratio[1]

        self.visible_categories = sorted_categories[:visible_categories_count]
        self.invisible_categories = sorted_categories[visible_categories_count:visible_categories_count + invisible_categories_count]

        print(f'self.visible_categories={self.visible_categories}')
        print(f'self.invisible_categories={self.invisible_categories}')


        visible_indices, invisible_indices = [], []
        for i in range(num_samples):


            if any(self.labels[i, cat] == 1 for cat in self.visible_categories):
                visible_indices.append(i)
            elif any(self.labels[i, cat] == 1 for cat in self.invisible_categories):
                invisible_indices.append(i)



        np.random.seed(2025)
        np.random.shuffle(visible_indices)
        np.random.shuffle(invisible_indices)
        print(f'visible_counts: {len(visible_indices)}')
        print(f'invisible_counts: {len(invisible_indices)}')

        self.visible_set = Subset(self, visible_indices)
        self.invisible_set = Subset(self, invisible_indices)

    def create_retrieval_and_query_sets(self):

        self.visible_retrieval_set, self.visible_query_set = self.create_visible_retrieval_and_query()


        self.invisible_retrieval_set, self.invisible_query_set = self.create_invisible_retrieval_and_query()

    def create_visible_retrieval_and_query(self):


        visible_indices = self.visible_set.indices if isinstance(self.visible_set, Subset) else np.arange(len(self.visible_set))

        query_indices = []
        retrieval_indices = []

        split_idx = len(visible_indices) // 10
        if split_idx > 2000:
            split_idx = 2000


        query_indices = visible_indices[:split_idx]
        retrieval_indices = visible_indices[split_idx:]


        print(f'visible_retrieval_counts: {len(retrieval_indices)}')
        print(f'visible_query_counts: {len(query_indices)}')
        return Subset(self, retrieval_indices), Subset(self, query_indices)

    def create_invisible_retrieval_and_query(self):

        invisible_indices = self.invisible_set.indices if isinstance(self.invisible_set, Subset) else np.arange(len(self.invisible_set))

        query_indices = []
        retrieval_indices = []


        split_idx = len(invisible_indices) // 10
        if split_idx > 2000:
            split_idx = 2000

        query_indices = invisible_indices[:split_idx]
        retrieval_indices = invisible_indices[split_idx:]
        print(f'invisible_retrieval_counts: {len(retrieval_indices)}')
        print(f'invisible_query_counts: {len(query_indices)}')
        return Subset(self, retrieval_indices), Subset(self, query_indices)


    def __getitem__(self, index):
        index = int(index)

        image = self.imgs[index]
        text = self.texts[index]
        if isinstance(self.imgs, Sampler):
            multi_crops = list(map(lambda trans: trans(image), self.trans))
            text = list(map(lambda trans: trans(text), [text_transform] * len(self.trans)))
        else:
            multi_crops = [image]
            text = [text]

        label = self.labels[index]
        return index, multi_crops, text, label

    def __len__(self):
        return len(self.imgs)


def MIRFlickr25K():
    root = 'MIRFLICKR25K'
    imgs = h5py.File('D:\Learning\datasets\mirflickr25K_raw_mat/mirflickr25k-iall.mat', mode='r')['IAll'][()]  # (20015, 3, 224, 224)
    tags = sio.loadmat('D:\Learning\datasets\mirflickr25K_raw_mat/mirflickr25k-yall.mat')['YAll']
    labels = sio.loadmat('D:\Learning\datasets\mirflickr25K_raw_mat/mirflickr25k-lall.mat')['LAll']

    return imgs.transpose([0, 3, 2, 1]), tags, labels, root

def MIRFlickr25K_fea():
    root = 'D:\Datasets\mirflickr25K_fea_mat'
    data_img = sio.loadmat(os.path.join(root, 'mirflickr25k-iall-vgg.mat'))['XAll']
    data_txt = sio.loadmat(os.path.join(root, 'mirflickr25k-yall.mat'))['YAll']
    labels = sio.loadmat(os.path.join(root, 'mirflickr25k-lall.mat'))['LAll']
    return data_img, data_txt, labels

def NUSWIDE():
    root = r'D:\Datasets\NUS-WIDE-TC10'
    imgs = h5py.File(os.path.join(root, 'nus-wide-tc10-iall.mat'), mode='r')['IAll'][()]
    tags = sio.loadmat(os.path.join(root, 'nus-wide-tc10-yall.mat'))['YAll']
    labels = sio.loadmat(os.path.join(root, 'nus-wide-tc10-lall.mat'))['LAll']
    return imgs.transpose([0, 3, 2, 1]), tags, labels, root

def NUSWIDE_fea():
    root = r'D:\Datasets\NUS-WIDE-TC10'
    data_img = sio.loadmat(os.path.join(root, 'nus-wide-tc10-xall-vgg.mat'))['XAll']
    data_txt = sio.loadmat(os.path.join(root, 'nus-wide-tc10-yall.mat'))['YAll']
    labels = sio.loadmat(os.path.join(root, 'nus-wide-tc10-lall.mat'))['LAll']
    return data_img, data_txt, labels

def MSCOCO_fea():
    root = r'D:\Datasets/'
    path = root + 'MSCOCO_deep_doc2vec_data_rand.h5py'
    data = h5py.File(path)
    data_img = data['XAll'][()]
    data_txt = data['YAll'][()]
    labels = data['LAll'][()]
    return data_img, data_txt, labels
