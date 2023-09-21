from torch.utils.data import Dataset
from typing import Callable
from torchvision import transforms as T
import numpy as np
import torch
import os
import cv2
from torchvision.transforms import functional as F
from scipy.ndimage.interpolation import zoom
from torchvision import transforms
import random
from scipy import ndimage
import glob
import SimpleITK as sitk
import tifffile as tiff
def LoadDatasets(A_path,val_A_path,training = True):
    img_size = 224

    A_DATA = ImageToImage2D(A_path, None, image_size=img_size,training = training)

    val_A = ImageToImage2D(val_A_path, None, image_size=img_size,training = training)

    return A_DATA,val_A

class ImageToImage2D(Dataset):
    """
    Reads the images and applies the augmentation transform on them.
    Usage:
        1. If used without the unet.model.Model wrapper, an instance of this object should be passed to
           torch.utils.data.DataLoader. Iterating through this returns the tuple of image, mask and image
           filename.
        2. With unet.model.Model wrapper, an instance of this object should be passed as train or validation
           datasets.

    Args:
        dataset_path: path to the dataset. Structure of the dataset should be:
            dataset_path
              |-- images
                  |-- img001.png
                  |-- img002.png
                  |-- ...
              |-- masks
                  |-- img001.png
                  |-- img002.png
                  |-- ...

        joint_transform: augmentation transform, an instance of JointTransform2D. If bool(joint_transform)
            evaluates to False, torchvision.transforms.ToTensor will be used on both image and mask.
        one_hot_mask: bool, if True, returns the mask in one-hot encoded form.
    """

    def __init__(self, dataset_path: str, joint_transform: Callable = None, one_hot_mask: int = False, image_size: int =224, training: int = True) -> None:
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.CT_path = os.path.join(dataset_path, 'CT')
        self.SUV_path = os.path.join(dataset_path, 'SUV')
        self.output_path = os.path.join(dataset_path, 'label')
        self.images_list = os.listdir(self.CT_path)
        self.one_hot_mask = one_hot_mask
        if training:
            if joint_transform:
            #self.joint_transform = joint_transform
                self.joint_transform = lambda x, y, z: (x, y, z)
            else:
                to_tensor = T.ToTensor()
                self.joint_transform = lambda x, y, z: (to_tensor(x), to_tensor(y), to_tensor(z))
        else:
            if joint_transform:
                # self.joint_transform = joint_transform
                self.joint_transform = lambda x, y: (x, y)
            else:
                to_tensor = T.ToTensor()
                self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

        self.rot_transform = transforms.Compose([RandomGenerator()])

    def __len__(self):
        return len(os.listdir(self.CT_path))

    def __getitem__(self, idx):

        image_filename = self.images_list[idx]
        #CT_image = cv2.imread(os.path.join(self.CT_path, image_filename))

        sitk_t1 = sitk.ReadImage(os.path.join(self.CT_path, image_filename))
        CT_image = np.squeeze(sitk.GetArrayFromImage(sitk_t1))


        sitk_t2 = sitk.ReadImage(os.path.join(self.SUV_path, image_filename[:-3]+'nii'))
        SUV_image = np.squeeze(sitk.GetArrayFromImage(sitk_t2))

        mask = cv2.imread(os.path.join(self.output_path, image_filename[:-3]+'png'),0)
        mask[mask>0]=1
        # correct dimensions if needed
        CT_image_float = CT_image.astype('float32')
        SUV_image_float = SUV_image.astype('float32')
        mask_float = mask.astype('float32')
        sample = {'CT_image': CT_image_float, 'SUV_image':SUV_image_float,'label': mask_float}
        rot_image, rot_t, rot_mask = self.rot_transform(sample)

        correct_dims_image, correct_dims_t_image, correct_dims_mask = correct_dims(rot_image, rot_t, rot_mask)

        #image, mask = F.to_pil_image(image), F.to_pil_image(mask)
        #sample = {'image': image, 'label': mask}

        if self.joint_transform:
            transform_image,transform_SUV_image,transform_mask = self.joint_transform(correct_dims_image,correct_dims_t_image, correct_dims_mask)
        #trans = transforms.ToTensor()
        # trans_image = trans(correct_dims_image)
        # trans_mask = trans(correct_dims_mask)

        sample = {'CT_image': transform_image, 'SUV_image': transform_SUV_image,'label': transform_mask}
        #rot_sample = self.rot_transform(sample)
        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        return sample, image_filename


def correct_dims(*images):
    corr_images = []
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)

    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images

class ValGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image, label = F.to_pil_image(image), F.to_pil_image(label)
        x, y = image.size
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = F.to_tensor(image)
        label = to_long_tensor(label)
        sample = {'image': image, 'label': label}
        return sample

def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()

class RandomGenerator(object):
    def __call__(self, sample):
        image, t_img, label = sample['CT_image'], sample['SUV_image'], sample['label']
        #image = linear_scaling(image)
        #image, label = F.to_pil_image(image), F.to_pil_image(label)
        #x, y = image.size
        if random.random() > 0.6:
            image, t_img, label = random_rot_flip(image, t_img, label)
        elif random.random() < 0.3:
            image, t_img, label = random_rotate(image, t_img, label)
        else:
            # 对比度增强
            image = image
            #image = np.clip(10 * (image - 128) + 128, -1024, 1024)


        x= image.shape[1]
        y = image.shape[1]
        return image, t_img, label

def random_rot_flip(image, t_img, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    t_img = np.rot90(t_img, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    t_img = np.flip(t_img, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, t_img, label

def random_rotate(image, t_img, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    t_img = ndimage.rotate(t_img, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, t_img, label


def normalize(data):
    """
    将NumPy数组归一化到[0, 1]范围内
    """
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val == min_val:
        return np.zeros(data.shape)
    else:
        return (data - min_val) / (max_val - min_val)
