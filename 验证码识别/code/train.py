# coding=utf-8
import time

from torch.optim import lr_scheduler
from tqdm import tqdm, trange
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import MyDataset
from model import Model
from one_hot import vec2text
import data_generator

captcha_list = data_generator.captcha_list
captcha_size = data_generator.captcha_size


def WriteData(f_name, *args):
    with open(f_name, 'a+') as f:
        for data in args:
            f.write(str(data)+"\t")
        f.write("\n")

def train(dataloader, model, loss_fn, optimizer,device):
    print('train:')
    size = len(dataloader.dataset)
    avg_loss = 0
    # 从数据加载器中读取batch（一次读取多少张，即批次数），X(图片数据)，y（图片真实标签）。
    for X, y in tqdm(dataloader):
        model.train()
        # 将数据存到显卡
        X, y = X.to(device), y.to(device)
        # 得到预测的结果pred
        pred = model(X)
        pred = pred.view(-1, captcha_size, captcha_list.__len__())
        # pred_text = vec2text(pred)
        # print(pred_text)
        # print(y.shape, pred.shape)

        loss = loss_fn(pred, y)
        avg_loss += loss
        # 反向传播，更新模型参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # break

    # 当一个epoch完了后返回平均 loss
    avg_loss /= size
    time.sleep(0.01)
    print(f"loss: {avg_loss}\n")
    avg_loss = avg_loss.detach().cpu().numpy()
    return avg_loss


def validate(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    # 将模型转为验证模式
    model.eval()
    # 初始化test_loss 和 correct， 用来统计每次的误差
    test_loss, correct = 0, 0
    # 测试时模型参数不用更新，所以no_gard()
    # 非训练， 推理期用到
    with torch.no_grad():
        print('val:')
        # 加载数据加载器，得到里面的X（图片数据）和y(真实标签）
        for X, y in tqdm(dataloader):
            # 将数据转到GPU
            X, y = X.to(device), y.to(device)
            y = y.view(-1, captcha_list.__len__())
            y_text = vec2text(y)
            # 将图片传入到模型当中就，得到预测的值pred
            pred = model(X)
            # print(pred)
            pred = pred.view(-1, captcha_list.__len__())
            pred_text = vec2text(pred)
            # 计算预测值pred和真实值y的差距
            test_loss += loss_fn(pred, y).item()
            # print(pred_text)
            # print(y_text)
            # break
            # 统计预测正确的个数(针对分类)
            for i in range(0, len(y_text), captcha_size):
                correct += pred_text[i:i+captcha_size] == y_text[i:i+captcha_size]
                # print(pred_text[i:i+captcha_size], y_text[i:i+captcha_size])
                # print(pred[i:i + captcha_size])
    test_loss /= size
    correct /= size
    time.sleep(0.01)
    print(f"Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss}")
    return correct, test_loss


if __name__ == '__main__':
    batch_size = 128

    # # 给训练集和测试集分别创建一个数据集加载器
    train_data = MyDataset("../data/train/")
    valid_data = MyDataset("../data/test/")


    train_dataloader = DataLoader(dataset=train_data, num_workers=4, pin_memory=True, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(dataset=valid_data, num_workers=4, pin_memory=True, batch_size=batch_size)

    # 如果显卡可用，则用显卡进行训练
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")


    model = Model(captcha_size * captcha_list.__len__())

    model = model.to(device)

    loss_fn = nn.MultiLabelSoftMarginLoss()

    # 定义优化器
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epochs = 25
    loss_ = 100
    save_root = "../models/"
    avg_loss = 0

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        start = time.time()
        avg_loss = train(train_dataloader, model, loss_fn, optimizer, device)
        # (dataloader, model, loss_fn, device)jif
        val_accuracy, val_loss = validate(valid_dataloader, model, loss_fn, device)

        end = time.time()
        print(f"The time it takes to run an epoch: {end - start} s\n")

        # 写入数据
        WriteData(save_root + "model.txt",
                  "epoch", t,
                  "train_loss", avg_loss,
                  "val_loss", val_loss,
                  "val_accuracy", val_accuracy)

        if avg_loss < loss_:
            loss_ = avg_loss
            torch.save(model.state_dict(), save_root + "model_best.pth")

        torch.save(model.state_dict(), save_root + "model_last.pth")