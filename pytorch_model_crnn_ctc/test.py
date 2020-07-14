import argparse
import cv2 as cv
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import os
from time import time

PICS_PATH = "../data/test"

CHARS = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
         "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
          "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新",
          "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B",
          "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P",
          "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "-"]

def parseOutput(output):
    label = ""
    last_char = ''
    last_is_char = -1  # 上一个char是‘-’时等于0，不是‘-’时等于1，初始值-1
    for i in range(output.shape[0]):
        latter = CHARS[output[i]]
        if latter == "-":
            last_is_char = 0
        else:
            if i > 0 and latter == last_char and last_is_char == 1:
                continue
            label += latter
            last_char = latter
            last_is_char = 1
    return label


class FeatureMap(torch.nn.Module):
    def __init__(self, batch):
        super(FeatureMap, self).__init__()
        self.batch = batch

    def forward(self, x):
        x = torch.split(x, 2, dim=3)
        tl = []
        for i in range(len(x)):
            tmp = x[i].reshape(self.batch, 32 * 8 * 2)
            tl.append(tmp)
        out = torch.stack(tl, dim=1)
        return out

class Net(torch.nn.Module):
    def __init__(self, batch, device, num_layers):
        super(Net, self).__init__()
        self.batch = batch
        self.device = device
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 32, 3, padding=1)
         # an affine operation: y = Wx + b
        self.num_layers = num_layers
        self.gru1 = nn.GRU(32*16, 128, num_layers=self.num_layers, bidirectional=True, dropout=0.3)
        self.fm = FeatureMap(self.batch)
        self.fc = nn.Linear(256, 66)
        #2*10  4*20 8*40 16*80 32*160

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.max_pool2d(x, (2, 2))
        x = self.fm(x)
        x = x.permute(1, 0, 2)
        x, h = self.gru1(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=2)
        return x

def main():
    pics = os.listdir(PICS_PATH)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = Net(1, device, 2).to(device)
    model.load_state_dict(torch.load("car_plate.pt"))
    model.eval()

    right_count = 0
    for i in range(len(pics)):
        label = pics[i][0:7]
        img = cv.imread(PICS_PATH + "/" + label + ".jpg")
        img = cv.resize(img, (160, 32))
        r, g, b = cv.split(img)
        numpy_array = np.array([r, g, b])

        img_tensor = torch.from_numpy(numpy_array)
        img_tensor = img_tensor.float()
        img_tensor /= 256
        img_tensor = img_tensor.reshape([1, 3, 32, 160])
        img_tensor = img_tensor.to(device)
        t1 = time()
        output = model(img_tensor).cpu()
        output = torch.squeeze(output)
        values, indexs = output.max(1)
        t2 = time()
        output_label = parseOutput(indexs)
        if output_label == label:
            right_count += 1
        print("label is " + label + " ,network predict is " + output_label+" cost is "+str(t2-t1)+"s")
    print(len(pics), right_count)
    print("correct rate is " + str(right_count / len(pics)))


if __name__ == '__main__':
    main()
