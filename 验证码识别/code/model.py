# coding=utf-8
#
# import torch
# from torch import nn
# import data_generator
#
# captcha_list = data_generator.captcha_list
# captcha_size = data_generator.captcha_size
#
# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)
#         )   # [64, 128, 30, 80]
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)
#         )   # [64, 128, 15, 40]
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)
#         )   # [64, 256, 7, 20]
#         self.layer4 = nn.Sequential(
#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)
#         )   # [64, 512, 3, 10]
#         self.layer5 = nn.Sequential(
#             nn.Flatten(),   # [64, 15360]
#             nn.Linear(15360, 4096),
#             nn.Dropout(0.3),
#             nn.ReLU(),
#             nn.BatchNorm1d(4096),
#             nn.Linear(4096, captcha_size * captcha_list.__len__())
#         )
#
#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.layer5(x)
#         return x
#
#
# if __name__ == '__main__':
#     data = torch.ones((64, 1, 60, 160))
#     model = Model()
#     x = model(data)
#     print(x)
#     print(x.shape)


import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, BatchNorm2d, ReLU, AdaptiveAvgPool2d

import torch
import data_generator

captcha_list = data_generator.captcha_list
captcha_size = data_generator.captcha_size

class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        self.model0 = Sequential(
            # 0
            # 输入3通道、输出64通道、卷积核大小、步长、补零、
            Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7), stride=2, padding=3),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
        )
        self.model1 = Sequential(
            # 1.1
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
        )

        self.R1 = ReLU()

        self.model2 = Sequential(
            # 1.2
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
        )

        self.R2 = ReLU()

        self.model3 = Sequential(
            # 2.1
            Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
        )
        self.en1 = Sequential(
            Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(128),
            ReLU(),
        )
        self.R3 = ReLU()

        self.model4 = Sequential(
            # 2.2
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
        )
        self.R4 = ReLU()

        self.model5 = Sequential(
            # 3.1
            Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
        )
        self.en2 = Sequential(
            Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(256),
            ReLU(),
        )
        self.R5 = ReLU()

        self.model6 = Sequential(
            # 3.2
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
        )
        self.R6 = ReLU()

        self.model7 = Sequential(
            # 4.1
            Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
        )
        self.en3 = Sequential(
            Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(512),
            ReLU(),
        )
        self.R7 = ReLU()

        self.model8 = Sequential(
            # 4.2
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
        )
        self.R8 = ReLU()

        # AAP 自适应平均池化
        self.aap = AdaptiveAvgPool2d((1, 1))
        # flatten 维度展平
        self.flatten = Flatten(start_dim=1)
        # FC 全连接层
        self.fc = Linear(512, num_classes)

    def forward(self, x):
        x = self.model0(x)

        f1 = x
        x = self.model1(x)
        x = x + f1
        x = self.R1(x)

        f1_1 = x
        x = self.model2(x)
        x = x + f1_1
        x = self.R2(x)

        f2_1 = x
        f2_1 = self.en1(f2_1)
        x = self.model3(x)
        x = x + f2_1
        x = self.R3(x)

        f2_2 = x
        x = self.model4(x)
        x = x + f2_2
        x = self.R4(x)

        f3_1 = x
        f3_1 = self.en2(f3_1)
        x = self.model5(x)
        x = x + f3_1
        x = self.R5(x)

        f3_2 = x
        x = self.model6(x)
        x = x + f3_2
        x = self.R6(x)

        f4_1 = x
        f4_1 = self.en3(f4_1)
        x = self.model7(x)
        x = x + f4_1
        x = self.R7(x)

        f4_2 = x
        x = self.model8(x)
        x = x + f4_2
        x = self.R8(x)

        # 最后3个
        x = self.aap(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    # 10分类
    data = torch.ones((1, 1, 60, 160))
    model = Model(captcha_size * captcha_list.__len__())
    x = model(data)
    print(x)
    print(x.shape)
