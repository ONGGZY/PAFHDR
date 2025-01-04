from yacs.config import CfgNode as CN

_C = CN()
_C.Train = CN()
_C.Val = CN()
_C.Optim = CN()
_C.Scheduler = CN()
_C.Resume = CN()

_C.Params =CN()
_C.Params.in_channels= 32
_C.Params.d_model = 192
_C.Params.out_channels=48
_C.Params.crop_size= 128
_C.Params.window_size = 8
_C.Params.num_layers= [1, 1, 1]


_C.Project_Name = 'MamHDR'
_C.Model_Name = 'MamHDR666'
_C.Data_name = 'Kalantari'
_C.Grad_accumulation_num = 1

_C.Resume.is_resume = False
_C.Resume.resume_best = False
_C.Resume.start_epoch = -1
_C.Resume.dir_checkpoints = './checkpoints'
_C.Resume.best_model='./checkpoints/best_model.pth'

_C.Train.dataset_root = '/Datasets/SIGGRAPH17_HDR/Training'
_C.Train.batch_size = 2

_C.Val.dataset_root = '/Datasets/SIGGRAPH17_HDR/Test'
_C.Val.batch_size = 1
_C.Val.val_per_epochs = 50

_C.Optim.lr_initial = 2e-4
_C.Optim.lr_min = 2e-6

_C.Scheduler.total_epochs = 5000
_C.Scheduler.warmup_epochs = 20
_C.Scheduler.warmup_lr_initial = 2e-6


def get_cfg_defaults():
    return _C.clone()
