import numpy as np
import torch
import torch.utils.data as Data
import nibabel as nb
import os, sys, glob
import SimpleITK as sitk


def load_4D(name):
    resamplng_shape = (128, 128, 128)
    X_nb = nb.load(name)
    X_np = X_nb.dataobj

    min_pixel_value = np.min(X_np)  # getting minimum pixel value which is -1 in my case.
    model_np = np.full(shape=(128, 128, 128), fill_value=min_pixel_value)
    x_dim, y_dim, z_dim = X_np.shape
    x_ltail = (resamplng_shape[0] - x_dim) // 2
    y_ltail = (resamplng_shape[1] - y_dim) // 2
    z_ltail = (resamplng_shape[2] - z_dim) // 2

    x_rtail = resamplng_shape[0] - x_ltail
    y_rtail = resamplng_shape[1] - y_ltail
    z_rtail = resamplng_shape[2] - z_ltail
    model_np[x_ltail:x_rtail, y_ltail:y_rtail, z_ltail:z_rtail] = X_np[:, :, :]
    model_np = np.reshape(model_np, (1,) + model_np.shape)
    return model_np


def imgnorm(N_I, index1=0.0001, index2=0.0001):
    I_sort = np.sort(N_I.flatten())
    I_min = I_sort[int(index1 * len(I_sort))]
    I_max = I_sort[-int(index2 * len(I_sort))]
    N_I = 1.0 * (N_I - I_min) / (I_max - I_min)
    N_I[N_I > 1.0] = 1.0
    N_I[N_I < 0.0] = 0.0
    N_I2 = N_I.astype(np.float32)
    return N_I2


def Norm_Zscore(img):
    img = (img - np.mean(img)) / np.std(img)
    return img


def save_img(I_img, savename):
    I2 = sitk.GetImageFromArray(I_img, isVector=False)
    sitk.WriteImage(I2, savename)


class Dataset(Data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, names_t1, names_t2, iterations, norm=True):
        """Initialization"""
        self.names_t1 = names_t1
        self.names_t2 = names_t2
        self.norm = norm
        self.iterations = iterations

    def __len__(self):
        """Denotes the total number of samples"""
        return self.iterations

    def __getitem__(self, step):
        """Generates one sample of data"""
        img_A = load_4D(self.names_t1[step])
        img_B = load_4D(self.names_t2[step])

        if self.norm:
            return imgnorm(img_A), imgnorm(img_B)
        else:
            return img_A, img_B
