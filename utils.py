import numpy as np
import os, glob
import cv2
from math import log10
import time
import torch
import skimage.metrics as sk
import math
import torchvision.transforms as transforms
from bisect import bisect_right
import torch
import torch.nn as nn
import torch.nn.functional as F

def ReadExpoTimes(fileName):
    return np.power(2, np.loadtxt(fileName))


def list_all_files_sorted(folderName, extension=""):
    return sorted(glob.glob(os.path.join(folderName, "*" + extension)))


def ReadImages(fileNames):
    imgs = []
    for img_str in fileNames:
        img = cv2.imread(img_str, -1)

        # equivalent to im2single from Matlab
        img = np.float32(img)
        img = img / (2 ** 16)
        img.clip(0, 1)
        imgs.append(img)
    return np.array(imgs)

def ReadLabel(fileName):
    label = cv2.imread(os.path.join(fileName, 'HDRImg.hdr'), flags=cv2.IMREAD_UNCHANGED)
    label = label[:, :, [2, 1, 0]]
    return label

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
 ])

def ReadCropLabel(fileName):
    label = cv2.imread(os.path.join(fileName, 'label.hdr'), flags=cv2.IMREAD_UNCHANGED)
    label = label[:, :, [2, 1, 0]]
    return label

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
 ])

def ReadMs(fileName):
    # Ms = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
    Ms = cv2.imread(fileName, cv2.IMREAD_UNCHANGED)
    Ms = torch.from_numpy(Ms).unsqueeze(2).numpy()
    Ms = transform(Ms)  # tensor 归一化
    Ms = Ms.permute(1, 2, 0).numpy()
    return Ms

def LDR_to_HDR(imgs, expo, gamma):
    return (imgs ** gamma) / expo


def range_compressor(x):
    return (np.log(1 + 5000 * x)) / np.log(1 + 5000)


def range_compressor_tensor(x):
    const_1 = torch.from_numpy(np.array(1.0)).to(device=x.device)
    # const_1 = torch.from_numpy(np.array(1.0))

    const_5000 = torch.from_numpy(np.array(5000.0)).to(device=x.device)
    # const_1 = torch.from_numpy(np.array(1.0))
    # const_5000 = torch.from_numpy(np.array(5000.0))
    return (torch.log(const_1 + const_5000 * x)) / torch.log(const_1 + const_5000)

def reverse_tonemap(CompressedImage):
    return ((np.power(5001, CompressedImage)) - 1) / 5000

def psnr(x, target):
    sqrdErr = np.mean((x - target) ** 2)
    return 10 * log10(1/sqrdErr)

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += sk.peak_signal_noise_ratio(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

def batch_SSIM(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    SSIM = 0
    for i in range(Img.shape[0]):
        SSIM += sk.structural_similarity(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range, channel_axis=0)
    return (SSIM/Img.shape[0])

def compute_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 2.0 # input -1~1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

class SSIM():
    def __init__(self, range=1):
        self.range = range

    def __call__(self, img1, img2):
        if not img1.shape == img2.shape:
            raise ValueError("Input images must have the same dimensions.")
        if img1.ndim == 2:  # Grey or Y-channel image
            return self._ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(self._ssim(img1, img2))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return self._ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError("Wrong input image dimensions.")

    def _ssim(self, img1, img2):
        C1 = (0.01 * self.range) ** 2
        C2 = (0.03 * self.range) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        return ssim_map.mean()


def sobel_texture_cosine_similarity_multichannel(img1, img2):
    """
    使用 Sobel 梯度余弦相似度衡量多通道特征图的纹理相似性
    参数:
        img1, img2: 输入图像张量，形状为 [B, C, H, W]
        save_path1: Sobel 结果保存路径（图像 1）
        save_path2: Sobel 结果保存路径（图像 2）
    返回:
        similarity: 梯度纹理相似性（多通道余弦相似度平均值）
    """
    # Sobel 滤波器
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3).to(dtype=img1.dtype,device=img1.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3).to(dtype=img1.dtype,device=img1.device)

    # 提取1个通道的梯度
    grad1_x = F.conv2d(img1[0:1, :, :], sobel_x, padding=1)
    grad1_y = F.conv2d(img1[0:1, :, :], sobel_y, padding=1)
    grad2_x = F.conv2d(img2[0:1, :, :], sobel_x, padding=1)
    grad2_y = F.conv2d(img2[0:1, :, :], sobel_y, padding=1)

    # 梯度幅值
    grad1 = torch.sqrt(grad1_x ** 2 + grad1_y ** 2)
    grad2 = torch.sqrt(grad2_x ** 2 + grad2_y ** 2)

    # 梯度方向向量化
    grad1_vector = torch.cat((grad1_x.view(-1), grad1_y.view(-1)), dim=0)
    grad2_vector = torch.cat((grad2_x.view(-1), grad2_y.view(-1)), dim=0)

    # 计算余弦相似度
    dot_product = torch.dot(grad1_vector, grad2_vector)
    norm1 = torch.norm(grad1_vector)
    norm2 = torch.norm(grad2_vector)
    cosine_similarity = dot_product / (norm1 * norm2 + 1e-8)
    return 1-cosine_similarity

def batch_aligned_loss(imgs1,imgs2):
    aligend_loss = 0.
    B, C , W , H = imgs1.shape
    for i in range(B):
        img1,img2 = imgs1[i],imgs2[i]
        loss = sobel_texture_cosine_similarity_multichannel(img1,img2)
        aligend_loss+=loss
    
    return aligend_loss/B

class PSNR():
    def __init__(self, range=1):
        self.range = range

    def __call__(self, img1, img2):
        mse = np.mean((img1 - img2) ** 2)
        return 20 * math.log10(self.range / math.sqrt(mse))


class SSIM():
    def __init__(self, range=1):
        self.range = range

    def __call__(self, img1, img2):
        if not img1.shape == img2.shape:
            raise ValueError("Input images must have the same dimensions.")
        if img1.ndim == 2:  # Grey or Y-channel image
            return self._ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(self._ssim(img1, img2))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return self._ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError("Wrong input image dimensions.")

    def _ssim(self, img1, img2):
        C1 = (0.01 * self.range) ** 2
        C2 = (0.03 * self.range) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        return ssim_map.mean()
