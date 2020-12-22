import torch
import torchvision.transforms as transforms
from PIL import Image
import glob
import os

class FastQTMTDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.root_path = "./FastQTMTDataset/train/128x128/"
        self.image_paths = os.listdir(self.root_path)
        self.all_img_paths = []
        for image_path in self.image_paths:
            imgs = os.listdir(os.path.join(self.root_path, image_path))
            for img in imgs:
                self.all_img_paths.append(os.path.join(self.root_path, image_path, img))
        self.transform = transforms.Compose([
                # transforms.Resize(256),
                transforms.ToTensor(),
        ])
        self.len = len(self.all_img_paths) # num of data

    def __getitem__(self, index):
        x = Image.open(self.all_img_paths[index])
        label = self.all_img_paths[index].split('_')[6].split('.')[0] # POC_WxH_SPLIT
        return self.transform(x), int(label)

    def __len__(self):
        return self.len

### glob.glob보다 os.listdir이 빠르다
# root_path = "./FastQTMTDataset/train/128x128/**/*.png"
# image_paths = glob.glob(root_path)
# print(len(image_paths))
# count0 = 0
# count1 = 0
# for img_path in image_paths:
#     label = int(img_path.split('_')[6].split('.')[0]) # POC_WxH_SPLIT
#     if label == 0:
#         count0 += 1
#     elif label == 1:
#         count1 += 1
# print("Label0:", count0, "Label1:", count1)

# root_path = "./FastQTMTDataset/train/128x128/"
# image_paths = os.listdir(root_path)
# print(len(image_paths))
# count0 = 0
# count1 = 0
# for img_path in image_paths:
#     imgs = os.listdir(os.path.join(root_path, img_path))
#     for img in imgs:
#         label = int(img.split('_')[2].split('.')[0]) # POC_WxH_SPLIT
#         if label == 0:
#             count0 += 1
#         elif label == 1:
#             count1 += 1
#
# print("Label0:", count0, "Label1:", count1)