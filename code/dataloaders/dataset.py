import os
import cv2
import torch
import random
import json
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
from torchvision import transforms
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
import matplotlib.pyplot as plt
from PIL import Image

from .polyp_preprocess import *

class BaseDataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
        ops_weak=None,
        ops_strong=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            with open(self._base_dir + "/train_slices.list", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            with open(self._base_dir + "/val.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/data/slices/{}.h5".format(case), "r")
        else:
            h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), "r")
        image = h5f["image"][:]
        label = h5f["label"][:]
        sample = {"image": image, "label": label}
        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        sample["idx"] = idx
        return sample

class PolypDataset(Dataset):
    def __init__(self, ds_root, csv_root, split='train', labeled_ratio=None,
                transform=None, ops_weak=None, ops_strong=None,):
        super(PolypDataset, self).__init__()
        self.ds_root = ds_root
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        self.image_paths = self.get_frames_from_csv(
            os.path.join(csv_root, split+'_frames.json'))

        random.shuffle(self.image_paths)

        if labeled_ratio is not None and self.split == "train":
            labeled_num = int(len(self.image_paths) * labeled_ratio)
            self.image_paths = self.image_paths[:labeled_num]
        self.image_len = len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.ds_root, self.image_paths[idx])
        label_path = image_path.replace(".jpg", ".png").replace('Frame', 'GT')
        image_path = os.path.join(self.ds_root, self.image_paths[idx])
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('L')

        sample = {"image": image, "label": label}
        if self.transform != None:
            if self.split == "train" and None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        sample["idx"] = idx

        return sample

    def __len__(self):
        return self.image_len

    def get_frames_from_csv(self, csv_path):
        frame_paths = []
        frame_num = 0
        with open(csv_path) as csv:
            data_dict = json.load(csv)
        case_list = list(data_dict.keys())
        for case in case_list:
            frame_paths.extend(data_dict[case][0])
            frame_num += data_dict[case][1]
        assert len(frame_paths) == frame_num, 'len(frame_paths) != frame_num'
        return frame_paths

class PolypDataset_w_path(Dataset):
    def __init__(self, ds_root, csv_root, split='train', labeled_ratio=None,
                transform=None, ops_weak=None, ops_strong=None,):
        super(PolypDataset_w_path, self).__init__()
        self.ds_root = ds_root
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        self.image_paths = self.get_frames_from_csv(
            os.path.join(csv_root, split+'_frames.json'))

        random.shuffle(self.image_paths)

        if labeled_ratio is not None and self.split == "train":
            labeled_num = int(len(self.image_paths) * labeled_ratio)
            self.image_paths = self.image_paths[:labeled_num]
        self.image_len = len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.ds_root, self.image_paths[idx])
        label_path = image_path.replace(".jpg", ".png").replace('Frame', 'GT')
        image_path = os.path.join(self.ds_root, self.image_paths[idx])
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('L')

        sample = {"image": image, "label": label}
        if self.transform != None:
            if self.split == "train" and None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        sample["idx"] = idx
        sample['im_path'] = image_path

        return sample

    def __len__(self):
        return self.image_len

    def get_frames_from_csv(self, csv_path):
        frame_paths = []
        frame_num = 0
        with open(csv_path) as csv:
            data_dict = json.load(csv)
        case_list = list(data_dict.keys())
        for case in case_list:
            frame_paths.extend(data_dict[case][0])
            frame_num += data_dict[case][1]
        assert len(frame_paths) == frame_num, 'len(frame_paths) != frame_num'
        return frame_paths

trsf_train_image_224 = Compose_imglabel([
    Random_crop_Resize_image(7),
    Resize_image(224, 224),
    Random_horizontal_flip_image(0.5),
    Random_vertical_flip_image(0.5),
    toTensor_image(),
    Normalize_image(MEAN, STD)
])

trsf_valid_image_224 = Compose_imglabel([
    Resize_image(224, 224),
    toTensor_image(),
    Normalize_image(MEAN, STD)
])

def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
        return image, label
    else:
        return image


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def color_jitter(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)

    # s is the strength of color distortion.
    s = 1.0
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return jitter(image)


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}
        return sample


class WeakStrongAugment_polyp(object):
    """returns weakly and strongly augmented images

    Args:
        object (tuple): output size of network
    """

    def __init__(self, output_size):
        self.output_size = output_size
        self.resize = Resize_image(self.output_size[1], self.output_size[0])
        self.crop = Random_crop_Resize_image(7)
        self.hflip = Random_horizontal_flip_image(0.5)
        self.vflip = Random_vertical_flip_image(0.5)

        s = 1.0
        self.color_jitter = ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        self.totensor = toTensor_image()
        self.norm = Normalize_image(MEAN, STD)


    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image, label = self.resize(image, label)
        # weak augmentation is crop / flip
        image, label = self.crop(image, label)
        image, label = self.hflip(image, label)
        image_weak, label = self.vflip(image, label)
        # strong augmentation is color jitter
        image_strong, label = self.color_jitter(image_weak, label)
        # fix dimensions
        image_weak, label = self.totensor(image_weak, label)
        totensor_image = transforms.ToTensor()
        image_strong = totensor_image(image_strong)
        image_weak, label = self.norm(image_weak, label)
        image_strong, _ = self.norm(image_strong, label)
        # image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        # image_weak = torch.from_numpy(image_weak.astype(np.float32)).unsqueeze(0)
        # label = torch.from_numpy(label.astype(np.uint8))

        sample = {
            # "image": image,
            "image_weak": image_weak,
            "image_strong": image_strong,
            "label_aug": label,
        }
        return sample

class MultiAugment_polyp(object):
    """returns weakly (geometric) then strongly (color) augmented images

    Args:
        object (tuple): output size of network
    """

    def __init__(self, output_size):
        self.output_size = output_size
        self.resize = Resize_image(self.output_size[1], self.output_size[0])
        self.crop = Random_crop_Resize_image(7)
        self.hflip = Random_horizontal_flip_image(0.5)
        self.vflip = Random_vertical_flip_image(0.5)
        bcs = (0.2, 1.8)
        h = 0.2
        self.color_jitter = ColorJitter(bcs, bcs, bcs, h)
        self.totensor = toTensor_image()
        self.norm = Normalize_image(MEAN, STD)

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image, label = self.resize(image, label)
        # weak augmentation is crop / flip
        image, label = self.crop(image, label)
        image, label = self.hflip(image, label)
        image_weak, label = self.vflip(image, label)
        # strong augmentation is color jitter
        image_strong, label = self.color_jitter(image_weak, label)
        image_strong2, label = self.color_jitter(image_weak, label)


        # fix dimensions
        image_weak, label = self.totensor(image_weak, label)
        image_weak, label = self.norm(image_weak, label)

        totensor_image = transforms.ToTensor()
        image_strong = totensor_image(image_strong)
        image_strong2 = totensor_image(image_strong2)
        image_strong3 = totensor_image(image_strong3)

        image_strong, _ = self.norm(image_strong, label)
        image_strong2, _ = self.norm(image_strong2, label)
        image_strong3, _ = self.norm(image_strong3, label)
        # image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        # image_weak = torch.from_numpy(image_weak.astype(np.float32)).unsqueeze(0)
        # label = torch.from_numpy(label.astype(np.uint8))

        sample = {
            # "image": image,
            "image_weak": image_weak,
            "image_strong": image_strong,
            "image_strong2": image_strong2,
            "label_aug": label,
        }
        return sample
    
class WeakStrongAugment(object):
    """returns weakly and strongly augmented images

    Args:
        object (tuple): output size of network
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        # weak augmentation is rotation / flip
        image_weak, label = random_rot_flip(image, label)
        # strong augmentation is color jitter
        image_strong = color_jitter(image_weak).type("torch.FloatTensor")
        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image_weak = torch.from_numpy(image_weak.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        sample = {
            "image": image,
            "image_weak": image_weak,
            "image_strong": image_strong,
            "label_aug": label,
        }
        return sample

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)



class KvasirDataset(Dataset): 
    def __init__(self, ds_root, transform=None, save='False'):
        super(KvasirDataset, self).__init__()
        self.ds_root = ds_root
        self.transform = transform
        self.save = save

        self.image_paths = os.listdir(os.path.join(self.ds_root, 'images'))
        self.image_len = len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.ds_root, 'images', self.image_paths[idx])
        label_path = image_path.replace('images', 'masks')
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('L')

        sample = {"image": image, "label": label}
        sample['im_path'] = ''
        if self.transform != None:
            sample = self.transform(sample)
        if self.save == 'True':
            sample['im_path'] = image_path
        return sample

    def __len__(self):
        return self.image_len
    
class PolypGenDataset(Dataset): #/mnt/data-hdd/wa/dataset/Polyp/PolypGen
    def __init__(self, ds_root, transform=None, save='False'):
        super(PolypGenDataset, self).__init__()
        self.ds_root = ds_root
        self.transform = transform
        self.image_paths = []
        self.save = save

        data_c = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
        for dc in data_c:
            images = os.listdir(os.path.join(self.ds_root, 'data_'+dc, 'images_'+dc))
            self.image_paths.extend(self.ds_root + '/data_'+dc + '/images_'+ dc + '/' + image for image in images)

        self.image_len = len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = image_path.replace('images_C', 'masks_C')[:-4]+'_mask.jpg'
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('L')

        sample = {"image": image, "label": label}
        sample['im_path'] = ''
        if self.transform != None:
            sample = self.transform(sample)
        if self.save == 'True':
            sample['im_path'] = image_path
        return sample

    def __len__(self):
        return self.image_len