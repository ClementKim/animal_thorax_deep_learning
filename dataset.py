import os
import json
import torch
import multiprocessing

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class CustumDataset(Dataset):
    def __init__(self, json_list, img_list, transform=None):
        self.json_list = json_list
        self.img_list = img_list
        self.transform = transform

    def __len__(self):
        return len(self.json_list)

    def __getitem__(self, idx):
        json_path = os.path.join(os.getcwd(), 'sample data', '라벨링데이터', self.json_list[idx])
        img_path = os.path.join(os.getcwd(), 'sample data', '원천데이터', self.img_list[idx])

        with open(json_path, 'r') as f:
            json_data = json.load(f)

        label_info = json_data["annotations"][0]
        label = torch.tensor(label_info['points'], dtype=torch.float32).view(-1)

        img_data = Image.open(img_path).convert("RGB")

        if self.transform:
            img_data = self.transform(img_data)

        return img_data, label


img_list = os.listdir(os.path.join(os.getcwd(), 'sample data', '원천데이터'))
label_list = os.listdir(os.path.join(os.getcwd(), 'sample data', '라벨링데이터'))

if ".DS_Store" in label_list:
    label_list.remove(".DS_Store")

if ".DS_Store" in img_list:
    img_list.remove(".DS_Store")

label_list.sort()
img_list.sort()

if len(label_list) != len(img_list):
    print("error")
    exit(1)

for i in range(len(label_list)):
    if label_list[i].split(".")[0] != img_list[i].split(".")[0]:
        print("error")
        exit(2)

transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

dataset = CustumDataset(label_list, img_list, transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=multiprocessing.cpu_count())
