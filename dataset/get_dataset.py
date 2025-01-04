from torch.utils.data import Dataset, DataLoader
from utils import *
import albumentations as A


class Sig17TrainDataset(Dataset):
    def __init__(self, traindata_path,crop_size,transforms=None, noise=None, is_training=True):
        list = os.listdir(traindata_path)
        self.crop_size = crop_size
        self.image_list = []
        self.num = 0
        self.transforms = transforms
        self.noise = noise
        self.is_training = is_training
        for scene in range(len(list)):
            expo_path = os.path.join(traindata_path, list[scene], 'exposure.txt')
            file_path = list_all_files_sorted(os.path.join(traindata_path, list[scene]), '.tif')
            label_path = os.path.join(traindata_path, list[scene])
            self.image_list += [[expo_path, file_path, label_path]]
            self.num = self.num + 1

    def __getitem__(self, idx):
        expoTimes = ReadExpoTimes(self.image_list[idx][0])
        imgs = ReadImages(self.image_list[idx][1])
        label = ReadLabel(self.image_list[idx][2])

        if self.is_training:
            image = np.concatenate([imgs[0], imgs[1], imgs[2], label], axis=2)
            image = self.transforms(image=image)['image']
            imgs[0] = image[:, :, 0:3]
            imgs[1] = image[:, :, 3:6]
            imgs[2] = image[:, :, 6:9]
            label = image[:, :, 9:12]
        pre_img0 = LDR_to_HDR(imgs[0], expoTimes[0], 2.2)  # gamma取2.2
        pre_img1 = LDR_to_HDR(imgs[1], expoTimes[1], 2.2)
        pre_img2 = LDR_to_HDR(imgs[2], expoTimes[2], 2.2)
        output0 = np.concatenate((imgs[0], pre_img0), 2)  # numpy为WHC 2为C
        output1 = np.concatenate((imgs[1], pre_img1), 2)
        output2 = np.concatenate((imgs[2], pre_img2), 2)

        # argument
        crop_size = self.crop_size
        H, W, _ = imgs[0].shape  # 这里imgs为BWHC，读取图片HW的值
        x = np.random.randint(0, H - crop_size - 1)  # 0~(H-1)
        y = np.random.randint(0, W - crop_size - 1)
        im1 = output0[x:x + crop_size, y:y + crop_size, :].astype(np.float32).transpose(2, 0, 1)
        im2 = output1[x:x + crop_size, y:y + crop_size, :].astype(np.float32).transpose(2, 0, 1)
        im3 = output2[x:x + crop_size, y:y + crop_size, :].astype(np.float32).transpose(2, 0, 1)
        im4 = label[x:x + crop_size, y:y + crop_size, :].astype(np.float32).transpose(2, 0, 1)
        im1 = torch.from_numpy(im1)  # TENSOR
        im2 = torch.from_numpy(im2)
        im3 = torch.from_numpy(im3)
        im4 = torch.from_numpy(im4)

        sample = {'input1': im1, 'input2': im2, 'input3': im3, 'label': im4}

        return sample

    def __len__(self):
        return self.num


class Sig17ValDataset(Dataset):
    def __init__(self, val_data_path, noise=None):
        list = os.listdir(val_data_path)
        self.image_list = []
        self.num = 0
        self.noise = noise
        for scene in range(len(list)):
            expo_path = os.path.join(val_data_path, list[scene], 'exposure.txt')
            file_path = list_all_files_sorted(os.path.join(val_data_path, list[scene]), '.tif')
            label_path = os.path.join(val_data_path, list[scene])
            self.image_list += [[expo_path, file_path, label_path]]
            self.num = self.num + 1

    def __getitem__(self, idx):
        expoTimes = ReadExpoTimes(self.image_list[idx][0])
        imgs = ReadImages(self.image_list[idx][1])
        label = ReadLabel(self.image_list[idx][2])

        pre_img0 = LDR_to_HDR(imgs[0], expoTimes[0], 2.2)  # gamma取2.2
        pre_img1 = LDR_to_HDR(imgs[1], expoTimes[1], 2.2)
        pre_img2 = LDR_to_HDR(imgs[2], expoTimes[2], 2.2)
        output0 = np.concatenate((imgs[0], pre_img0), 2)  # numpy为WHC 2为C
        output1 = np.concatenate((imgs[1], pre_img1), 2)
        output2 = np.concatenate((imgs[2], pre_img2), 2)

        # argument
        crop_size = 128
        H, W, _ = imgs[0].shape  # 这里imgs为BWHC，读取图片HW的值
        x = np.random.randint(0, H - crop_size - 1)  # 0~(H-1)
        y = np.random.randint(0, W - crop_size - 1)
        im1 = output0[x:x + crop_size, y:y + crop_size, :].astype(np.float32).transpose(2, 0, 1)
        im2 = output1[x:x + crop_size, y:y + crop_size, :].astype(np.float32).transpose(2, 0, 1)
        im3 = output2[x:x + crop_size, y:y + crop_size, :].astype(np.float32).transpose(2, 0, 1)
        im4 = label[x:x + crop_size, y:y + crop_size, :].astype(np.float32).transpose(2, 0, 1)
        im1 = torch.from_numpy(im1)  # TENSOR
        im2 = torch.from_numpy(im2)
        im3 = torch.from_numpy(im3)
        im4 = torch.from_numpy(im4)

        sample = {'input1': im1, 'input2': im2, 'input3': im3, 'label': im4}

        return sample

    def __len__(self):
        return self.num

class Sig17CropTrainDataset(Dataset):
    def __init__(self, val_data_path, noise=None):
        list = os.listdir(val_data_path)
        self.image_list = []
        self.num = 0
        self.noise = noise
        for scene in range(len(list)):
            expo_path = os.path.join(val_data_path, list[scene], 'exposure.txt')
            file_path = list_all_files_sorted(os.path.join(val_data_path, list[scene]), '.tif')
            label_path = os.path.join(val_data_path, list[scene])
            self.image_list += [[expo_path, file_path, label_path]]
            self.num = self.num + 1

    def __getitem__(self, idx):
        expoTimes = ReadExpoTimes(self.image_list[idx][0])
        imgs = ReadImages(self.image_list[idx][1])
        label = ReadCropLabel(self.image_list[idx][2])

        pre_img0 = LDR_to_HDR(imgs[0], expoTimes[0], 2.2)  # gamma取2.2
        pre_img1 = LDR_to_HDR(imgs[1], expoTimes[1], 2.2)
        pre_img2 = LDR_to_HDR(imgs[2], expoTimes[2], 2.2)
        output0 = np.concatenate((imgs[0], pre_img0), 2)  # numpy为WHC 2为C
        output1 = np.concatenate((imgs[1], pre_img1), 2)
        output2 = np.concatenate((imgs[2], pre_img2), 2)

        # argument
        im1 = output0.astype(np.float32).transpose(2, 0, 1)
        im2 = output1.astype(np.float32).transpose(2, 0, 1)
        im3 = output2.astype(np.float32).transpose(2, 0, 1)
        im4 = label.astype(np.float32).transpose(2, 0, 1)
        im1 = torch.from_numpy(im1)  # TENSOR
        im2 = torch.from_numpy(im2)
        im3 = torch.from_numpy(im3)
        im4 = torch.from_numpy(im4)

        sample = {'input1': im1, 'input2': im2, 'input3': im3, 'label': im4}

        return sample

    def __len__(self):
        return self.num
class Sig17TestDataset(Dataset):
    def __init__(self, val_data_path, noise=None):
        list = os.listdir(val_data_path)
        self.image_list = []
        self.num = 0
        self.noise = noise
        for scene in range(len(list)):
            expo_path = os.path.join(val_data_path, list[scene], 'exposure.txt')
            file_path = list_all_files_sorted(os.path.join(val_data_path, list[scene]), '.tif')
            label_path = os.path.join(val_data_path, list[scene])
            self.image_list += [[expo_path, file_path, label_path]]
            self.num = self.num + 1

    def __getitem__(self, idx):
        expoTimes = ReadExpoTimes(self.image_list[idx][0])
        imgs = ReadImages(self.image_list[idx][1])
        label = ReadLabel(self.image_list[idx][2])

        pre_img0 = LDR_to_HDR(imgs[0], expoTimes[0], 2.2)  # gamma取2.2
        pre_img1 = LDR_to_HDR(imgs[1], expoTimes[1], 2.2)
        pre_img2 = LDR_to_HDR(imgs[2], expoTimes[2], 2.2)
        output0 = np.concatenate((imgs[0], pre_img0), 2)  # numpy为WHC 2为C
        output1 = np.concatenate((imgs[1], pre_img1), 2)
        output2 = np.concatenate((imgs[2], pre_img2), 2)

        # argument
        im1 = output0.astype(np.float32).transpose(2, 0, 1)
        im2 = output1.astype(np.float32).transpose(2, 0, 1)
        im3 = output2.astype(np.float32).transpose(2, 0, 1)
        im4 = label.astype(np.float32).transpose(2, 0, 1)
        im1 = torch.from_numpy(im1)  # TENSOR
        im2 = torch.from_numpy(im2)
        im3 = torch.from_numpy(im3)
        im4 = torch.from_numpy(im4)

        sample = {'input1': im1, 'input2': im2, 'input3': im3, 'label': im4}

        return sample

    def __len__(self):
        return self.num

if __name__ == '__main__':
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ])
    # train_path ='E:\\CV\\Datasets\\SIGGRAPH17_HDR\\Training'
    # train_data = Sig17TrainDataset(traindata_path=train_path, transforms=transform, noise=None,
    #                        is_training=True)
    # train_loader = DataLoader(train_data, batch_size=2, shuffle=True, num_workers=1,
    #                           drop_last=True, pin_memory=False)
    # for sample in train_loader:
    #     print(sample['input1'])
    #     break

    val_path = r'E:\CV\Datasets\SIGGRAPH17_HDR\Test'
    val_dataset = Sig17ValDataset(val_data_path=val_path)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    for data in val_loader:
        print(data['input1'].shape)
        break
