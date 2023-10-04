import random

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from torchvision import transforms
from torchvision.transforms import ToTensor as torchtotensor

# statistics from SUN-SEG dataset: https://github.com/GewelsJI/VPS/blob/main/lib/dataloader/statistics.pth
MEAN = np.array([0.4732661 , 0.44874457, 0.3948762 ], dtype=np.float32)
STD = np.array([0.22674961, 0.22012031, 0.2238305 ], dtype=np.float32)

class Compose_imglabel(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample["image"], sample["label"] = t(sample["image"], sample["label"])
        return sample

class Random_crop_Resize_image(object):
    def _randomCrop(self, img, label, x, y):
        width, height = img.size
        region = [x, y, width - x, height - y]
        img, label = img.crop(region), label.crop(region)
        img = img.resize((width, height), Image.BILINEAR)
        label = label.resize((width, height), Image.NEAREST)
        return img, label

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img, label):
        x, y = random.randint(0, self.crop_size), random.randint(0, self.crop_size)
        res_img, res_label = self._randomCrop(img, label, x, y)
        return res_img, res_label


class Random_horizontal_flip_image(object):
    def _horizontal_flip(self, img, label):
        return img.transpose(Image.FLIP_LEFT_RIGHT), label.transpose(Image.FLIP_LEFT_RIGHT)

    def __init__(self, prob):
        '''
        :param prob: should be (0,1)
        '''
        assert prob >= 0 and prob <= 1, "prob should be [0,1]"
        self.prob = prob

    def __call__(self, img, label):
        '''
        flip img and label simultaneously
        :param img:should be PIL image
        :param label:should be PIL image
        :return:
        '''
        if random.random() < self.prob:
            res_img, res_label = self._horizontal_flip(img, label)
            return res_img, res_label
        else:
            return img, label


class Random_vertical_flip_image(object):
    def _vertical_flip(self, img, label):
        return img.transpose(Image.FLIP_TOP_BOTTOM), label.transpose(Image.FLIP_TOP_BOTTOM)

    def __init__(self, prob):
        '''
        :param prob: should be (0,1)
        '''
        assert prob >= 0 and prob <= 1, "prob should be [0,1]"
        self.prob = prob

    def __call__(self, img, label):
        '''
        flip img and label simultaneously
        :param img:should be PIL image
        :param label:should be PIL image
        :return:
        '''
        if random.random() < self.prob:
            res_img, res_label = self._vertical_flip(img, label)
            return res_img, res_label
        else:
            return img, label

class Resize_image(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, img, label):
        res_img = img.resize((self.width, self.height), Image.BILINEAR)
        res_label = label.resize((self.width, self.height), Image.NEAREST)
        return res_img, res_label

import PIL.ImageFilter as PILFilter


class Dilate_label(object):
    def __init__(self, k_size=3):
        self.k_size = k_size

    def __call__(self, img, label):
        res_img = img
        res_label = label.filter(PILFilter.MaxFilter(self.k_size))
        return res_img, res_label


class Normalize_image(object):
    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def __call__(self, img, label):
        for i in range(3):
            img[:, :, i] -= float(self.mean[i])
        for i in range(3):
            img[:, :, i] /= float(self.std[i])
        return img, label

    
class toTensor_image(object):
    def __init__(self):
        self.totensor = torchtotensor()

    def __call__(self, img, label):
        res_img = self.totensor(img)
        res_label = self.totensor(label).long().squeeze(0)
        return res_img, res_label

class ColorJitter(object):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2):
        self.colorjitter = transforms.ColorJitter(brightness, contrast, saturation, hue)
        
    def __call__(self, img, label):
        img = self.colorjitter(img)
        return img, label

class RandomGaussianBlur(object):
    def __init__(self, radius=2):
        self._filter = GaussianBlur(radius=radius)

    def __call__(self, image, label):
        if random.random() < 0.5:
            image = self._filter(image)
        return image, label


class GaussianBlur(torch.nn.Module):
    def __init__(self, radius):
        super(GaussianBlur, self).__init__()
        self.radius = radius
        self.kernel_size = 2 * radius + 1
        self.sigma = 0.3 * (self.radius - 1) + 0.8
        self.kernel = torch.nn.Conv2d(3, 3, self.kernel_size, stride=1,
                                padding=self.radius, bias=False, groups=3)
        self.weight_init()

    def forward(self, input):
        assert input.size(1) == 3
        return self.kernel(input)

    def weight_init(self):
        weights = np.zeros((self.kernel_size, self.kernel_size))
        weights[self.radius, self.radius] = 1
        weight = gaussian_filter(weights, sigma=self.sigma)
        for param in self.kernel.parameters():
            param.data.copy_(torch.from_numpy(weight))
            param.requires_grad = False
