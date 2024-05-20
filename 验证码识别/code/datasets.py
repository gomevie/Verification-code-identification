# coding=utf-8
import numpy as np
import tensorflow
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms
from PIL import Image
from one_hot import text2vec, vec2text

class MyDataset(Dataset):
    def __init__(self, root_dir):
        super(MyDataset, self).__init__()
        self.image_path = [os.path.join(root_dir, image_name) for image_name in os.listdir(root_dir)]
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((60, 160)),
                transforms.Grayscale(),
                transforms.Normalize([0.5], [0.5])
            ]
        )

    def __len__(self):
        return self.image_path.__len__()

    def __getitem__(self, item):
        image_path = self.image_path[item]
        image = self.transforms(Image.open(image_path))
        label = image_path.split('/')[-1].split('_')[0]
        label_tensor = text2vec(label)
        return image, label_tensor



if __name__ == '__main__':
    data_dataset = MyDataset('../data/test/')
    image, label = data_dataset[0]

    print(image, label)
    print(label.shape)
    # valid_dataloader = DataLoader(dataset=data_dataset, num_workers=4, pin_memory=True, batch_size=1)
    # for X, y in valid_dataloader:
    #     print(X, y)
