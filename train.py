import os
import time
import wandb
import timm.scheduler
import torch.optim as optim
import torch
import numpy as np
import random
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.mam_hdr import MamHDR
from config import get_cfg_defaults
from losses import *
from pathlib import Path
from dataset.get_dataset import *
import albumentations

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)
torch.backends.cudnn.benchmark = True
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

cfg = get_cfg_defaults()
cfg.merge_from_file('configs/config.yml')
cfg.freeze()
print(cfg)


# --------Create_Model--------#
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_uniform_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:1")
in_channels = cfg.Params.in_channels
d_model = cfg.Params.d_model
out_channels = cfg.Params.out_channels
num_layers = cfg.Params.num_layers
crop_size = cfg.Params.crop_size
window_size = cfg.Params.window_size

net = MamHDR(in_channels, d_model, out_channels,num_layers,window_size).to(device)

net.apply(weights_init_normal)
print('total params:', sum([x.numel() for x in net.parameters()]))

# -------Optimizer---------#
lr_initial = cfg.Optim.lr_initial
lr_min = cfg.Optim.lr_min
optimizer = optim.Adam(net.parameters(), lr=lr_initial, betas=(0.9, 0.999), eps=1e-8)

# ----------Scheduler----------#
total_epochs = cfg.Scheduler.total_epochs
warmup_epochs = cfg.Scheduler.warmup_epochs
warmup_lr_initial = cfg.Scheduler.warmup_lr_initial
scheduler = timm.scheduler.CosineLRScheduler(optimizer=optimizer,
                                             t_initial=total_epochs,
                                             lr_min=lr_min,
                                             warmup_t=warmup_epochs,
                                             warmup_lr_init=warmup_lr_initial
                                             )
# ---------Loss-----------#
L1_loss = nn.L1Loss()
Get_gradient = Get_gradient()
GC_Loss = GradientCosineLoss()

# --------Resume--------#
def load_checkpoint(checkpoint, model, optimizer):
    checkpoint = torch.load(checkpoint)
    new_state_dict = {}
    for key, value in checkpoint['model_state'].items():
        new_key = key[7:] if key.startswith('module.') else key
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)
    optimizer.load_state_dict(checkpoint['optim_state'])
    return checkpoint['epoch']


dir_checkpoints = Path(cfg.Resume.dir_checkpoints)
start_epoch = 0
if cfg.Resume.is_resume:
    if cfg.Resume.resume_best:
        checkpoint = cfg.Resume.best_model
        start_epoch = load_checkpoint(checkpoint, net, optimizer) + 1
        print('Successfully load best_model.pth')
    elif cfg.Resume.start_epoch > 0:
        start_epoch = cfg.Resume.start_epoch
        print(str(dir_checkpoints / "checkpoint_epoch{}.pth".format(start_epoch)))
        checkpoint = str(dir_checkpoints / "checkpoint_epoch{}.pth".format(start_epoch))
        start_epoch = load_checkpoint(checkpoint, net, optimizer) + 1
        print('Successfully checkpoint_epoch{}.pth'.format(start_epoch))

# ------------Dist--------------#
# if len(device_ids) > 1:
#     net = nn.DataParallel(net, device_ids=device_ids)
#     print(f'Using {len(device_ids)} GPUs')

# -----------DataLoader----------------#
train_dir = cfg.Train.dataset_root
val_dir = cfg.Val.dataset_root
train_batch_size = cfg.Train.batch_size
val_batch_size = cfg.Val.batch_size

transforms = albumentations.Compose([
    albumentations.HorizontalFlip(p=0.5),  # 水平翻转，翻转概率为0.5
    albumentations.VerticalFlip(p=0.5),
])

assert os.path.exists(train_dir), f'{train_dir} does not exist'
train_dataset = Sig17TrainDataset(train_dir, crop_size, transforms=transforms)
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=16, drop_last=False,
                          pin_memory=True)

assert os.path.exists(val_dir), f'{val_dir} does not exist'
val_dataset = Sig17TestDataset(val_dir)
val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=16, pin_memory=True)

# ---------------wandb-----------------#
model_Name = cfg.Model_Name + '-' + str(in_channels) + '-' + str(num_layers) + '-' + str(
    d_model)+'-'+str(out_channels)
wandb.init(project=cfg.Project_Name, name=model_Name, config={'lr_min': lr_min,
                                                              'lr_initial': lr_initial,
                                                              'warmup_lr_initial': warmup_lr_initial,
                                                              'warmup_epochs': warmup_epochs,
                                                              'epochs': total_epochs,
                                                              'batch_size': cfg.Train.batch_size,
                                                              'Data_Name': cfg.Data_name
                                                              })

# --------Start!!-----------#
print(f'start_epoch:{start_epoch + 1}')
val_per_epochs = cfg.Val.val_per_epochs
gra_ac_num = cfg.Grad_accumulation_num
best_PSNR_u = 0.0
best_epoch = 1
for epoch in range(start_epoch, total_epochs):
    train_loss = 0.0
    train_PSNR = 0.0
    train_SSIM = 0.0
    train_PSNR_u = 0.0
    train_SSIM_u = 0.0
    iter_num = 0
    scheduler.step(epoch)
    net.train()
    optimizer.zero_grad()
    for i, sample in enumerate(tqdm(train_loader)):
        img1, img2, img3, labels = sample['input1'], sample['input2'], sample['input3'], sample['label']
        img1, img2, img3, labels = img1.to(device), img2.to(device), img3.to(device), labels.to(device)
        out, aligned_loss = net(img1, img2, img3)

        # reslut1_map,r1_maph,r1_mapv=Get_gradient(result1)
        label_gradient_map,label_maph,label_mapv = Get_gradient(labels)
        out_gradient_map,_ ,_ = Get_gradient(out)

        # aligned_loss = 0.4*L1_loss(label_gradient_map,reslut1_map)+0.6*GC_Loss(r1_maph,r1_mapv,label_maph,label_mapv)

        PSNR = batch_PSNR(torch.clamp(out, 0., 1.), labels, 1.)
        train_PSNR += PSNR.item()

        SSIM = batch_SSIM(torch.clamp(out, 0., 1.), labels, 1.)
        train_SSIM += SSIM.item()

        pre_compressed = range_compressor_tensor(out)
        label_compressed = range_compressor_tensor(labels)

        PSNR_u = batch_PSNR(torch.clamp(pre_compressed, 0., 1.), label_compressed, 1.)
        train_PSNR_u += PSNR_u.item()

        SSIM_u = batch_SSIM(torch.clamp(pre_compressed, 0., 1.), label_compressed, 1.)
        train_SSIM_u += SSIM_u.item()
        total_loss = 0.5*aligned_loss + L1_loss(out, labels) + 0.5*L1_loss(pre_compressed, label_compressed) + 0.5*L1_loss(out_gradient_map,label_gradient_map)+0.2 * (1 - SSIM)+0.2*(1-SSIM_u)

        train_loss += total_loss.item()
        total_loss.backward()
        if (i + 1) % gra_ac_num == 0:
            optimizer.step()
            optimizer.zero_grad()
        iter_num += 1
        # print(f"{PSNR:.2f}\t{PSNR_u1:.2f}\t{SSIM_u1:.4f}\t{SSIM_u1:.4f}\t{total_loss:.4f}")
    lr = optimizer.param_groups[0]['lr']
    train_loss, train_PSNR, train_PSNR_u, train_SSIM, train_SSIM_u = train_loss / iter_num, train_PSNR / iter_num, train_PSNR_u / iter_num, train_SSIM / iter_num, train_SSIM_u / iter_num
    print(
        "Epoch: {}/{}\tlr {:.6f}\tepoch_loss: {:.4f}\nPSNR: {:.4}\tPSNR_u: {:.4}\tSSIM: {:.4}\tSSIM_u: {:.4}".format(
            epoch + 1, total_epochs,
            lr,
            train_loss,
            train_PSNR,
            train_PSNR_u,
            train_SSIM,
            train_SSIM_u
        ))
    wandb.log({'epoch': epoch + 1, 'train_loss': train_loss, 'lr': lr, "train_PSNR": train_PSNR,
               "train_PSNR_u": train_PSNR_u,
               "train_SSIM": train_SSIM, "train_SSIM_u": train_SSIM_u})

    if (epoch + 1) % val_per_epochs == 0:
        val_PSNR = 0.
        val_PSNR_u = 0.
        val_SSIM = 0.
        val_SSIM_u = 0.
        test_iter_num = 0
        net.eval()
        print('After {} epochs, evaluating on validation ...'.format(epoch + 1))
        for sample in tqdm(val_loader):
            val_imgs1, val_imgs2, val_imgs3, val_labels = sample['input1'], sample['input2'], sample['input3'], \
                sample['label']
            val_imgs1, val_imgs2, val_imgs3, val_labels = val_imgs1.to(device), val_imgs2.to(
                device), val_imgs3.to(device), val_labels.to(device)
            with torch.no_grad():
                val_out,val_aligned = net(val_imgs1, val_imgs2, val_imgs3)

                val_PSNR += batch_PSNR(torch.clamp(val_out, 0., 1.), val_labels, 1.).item()

                val_SSIM += batch_SSIM(torch.clamp(val_out, 0., 1.), val_labels, 1.).item()

                val_pre_compressed = range_compressor_tensor(val_out)
                val_label_compressed = range_compressor_tensor(val_labels)

                val_PSNR_u += batch_PSNR(torch.clamp(val_pre_compressed, 0., 1.), val_label_compressed, 1.).item()

                val_SSIM_u += batch_SSIM(torch.clamp(val_pre_compressed, 0., 1.), val_label_compressed, 1.).item()
            test_iter_num += 1
        val_PSNR, val_PSNR_u, val_SSIM, val_SSIM_u = val_PSNR / test_iter_num, val_PSNR_u / test_iter_num, val_SSIM / test_iter_num, val_SSIM_u / test_iter_num

        if val_PSNR_u >= best_PSNR_u:
            best_PSNR_u = val_PSNR_u
            best_epoch = epoch + 1
            Path(dir_checkpoints).mkdir(parents=True, exist_ok=True)
            torch.save({'epoch': epoch,
                        'model_state': net.state_dict(),
                        'optim_state': optimizer.state_dict()
                        }, str(dir_checkpoints / "best_model.pth"))
        print(
            "val_PSNR: {:.4f}\tval_PSNR_u: {:.4f}\tval_SSIM: {:.4f}\tval_SSIM_u: {:.4f}\tbest_PSNR_u: {:.4f}\tbest_epoch: {}".format(
                val_PSNR, val_PSNR_u,
                val_SSIM, val_SSIM_u, best_PSNR_u, best_epoch
            ))
        wandb.log({'epoch_val': epoch + 1, "val_PSNR": val_PSNR, "val_PSNR_u": val_PSNR_u,
                   "val_SSIM": val_SSIM, "val_SSIM_u": val_SSIM_u})
    if (epoch + 1) % 100 == 0 and (epoch + 1) >= 2500:
        torch.save({'epoch': epoch,
                    'model_state': net.state_dict(),
                    'optim_state': optimizer.state_dict()
                    }, str(dir_checkpoints / "checkpoint_epoch{}.pth".format(epoch + 1)))
        print('Successfully to save {} state_dict.'.format(epoch + 1))
wandb.fish()
