import os
import numpy
import numpy as np
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.mam_hdr import MamHDR
from dataset.get_dataset import Sig17TestDataset
from utils import *


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# os.environ['CUDA_VISIBLE_DEVICES'] = "1"

def load_checkpoint(checkpoint, model):
    checkpoint = torch.load(checkpoint)
    new_state_dict = {}
    for key, value in checkpoint['model_state'].items():
        new_key = key[7:] if key.startswith('module.') else key
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)
    return model


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:1")
print(device)
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='./checkpoints/checkpoint_epoch5500.pth', type=str)
parser.add_argument('--test_dir', default='/703_files/gzy/SIG/Test', type=str)
args = parser.parse_args()

checkpoint = args.checkpoint
test_dir = args.test_dir

net = MamHDR(64,64,64,[6,6,6],64).to(device)
load_checkpoint(checkpoint, net)
print(f'successfully loaded {checkpoint}')
print('total params:', sum([x.numel() for x in net.parameters()]))

assert os.path.exists(test_dir), f'{test_dir} does not exist'
test_dataset = Sig17TestDataset(test_dir)
test_loader = DataLoader(test_dataset, 1, shuffle=False, num_workers=10, pin_memory=True)

device_ids = [i for i in range(torch.cuda.device_count())]
# if len(device_ids) > 1:
#     net = nn.DataParallel(net, device_ids=device_ids)
#     print(f'Using {len(device_ids)} GPUs')

PSNR_sum = 0
PSNR_u_sum = 0
SSIM_sum = 0
SSIM_u_sum = 0
test_iter_num = 0
net.eval()
with torch.no_grad():
    for sample in test_loader:
        img1, img2, img3, label = sample['input1'], sample['input2'], sample['input3'], \
            sample['label']
        img1, img2, img3, label = img1.to(device), img2.to(device), img3.to(device), label.to(device)
        pre,_ = net(img1, img2, img3)
        pre_u = range_compressor_tensor(pre)
        label_u = range_compressor_tensor(label)

        PSNR_per = batch_PSNR(torch.clamp(pre, 0., 1.), label, 1.).item()
        PSNR_sum += PSNR_per

        SSIM_per = batch_SSIM(torch.clamp(pre, 0., 1.), label, 1.).item()
        SSIM_sum += SSIM_per

        PSNR_u_per = batch_PSNR(torch.clamp(pre_u, 0., 1.), label_u, 1.).item()
        PSNR_u_sum += PSNR_u_per

        SSIM_u_per = batch_SSIM(torch.clamp(pre_u, 0., 1.), label_u, 1.).item()
        SSIM_u_sum += SSIM_u_per

        print(f'PSNR_l:{PSNR_per:.4f}\tPSNR_u:{PSNR_u_per:.4f}\tSSIM_l:{SSIM_per:.4f}\tSSIM_u:{SSIM_u_per:.4f}')
        test_iter_num += 1

PSNR_l = PSNR_sum / test_iter_num
SSIM_l = SSIM_sum / test_iter_num
PSNR_u = PSNR_u_sum / test_iter_num
SSIM_u = SSIM_u_sum / test_iter_num
print(f'average: \tPSNR_l:{PSNR_l:.4f}\tPSNR_u:{PSNR_u:.4f}\tSSIM_l:{SSIM_l:.4f}\tSSIM_u:{SSIM_u:.4f}')
