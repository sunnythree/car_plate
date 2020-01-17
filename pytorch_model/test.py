import argparse
import cv2 as cv
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import os

PICS_PATH = "../data/test"

INDEX_PROVINCE = {"京": 0, "沪": 1, "津": 2, "渝": 3, "冀": 4, "晋": 5, "蒙": 6, "辽": 7, "吉": 8, "黑": 9, "苏": 10,
                  "浙": 11, "皖": 12, "闽": 13, "赣": 14, "鲁": 15, "豫": 16, "鄂": 17, "湘": 18, "粤": 19, "桂": 20,
                  "琼": 21, "川": 22, "贵": 23, "云": 24, "藏": 25, "陕": 26, "甘": 27, "青": 28, "宁": 29, "新": 30}

INDEX_LETTER = {"0": 31, "1": 32, "2": 33, "3": 34, "4": 35, "5": 36, "6": 37, "7": 38, "8": 39, "9": 40, "A": 41, "B": 42, "C": 43, "D": 44, "E": 45, "F": 46, "G": 47,"H": 48, "J": 49, "K": 50, "L": 51, "M": 52,
                "N": 53, "P": 54, "Q": 55, "R": 56, "S": 57, "T": 58, "U": 59, "V": 60, "W": 61, "X": 62, "Y": 63, "Z": 64}

PLATE_CHARS_PROVINCE = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
                        "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
                        "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新"]

PLATE_CHARS_LETTER = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P",
                      "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

def parseOutput(output):
    index = 0
    maxValue = 0
    label = ""
    output = output[0]
    for i in range(31):
        if output[i] > maxValue:
            maxValue = output[i]
            index = i
    label = label + PLATE_CHARS_PROVINCE[index]
    for j in range(6):
        index = 0
        maxValue = 0
        for i in range(34):
            if output[i+j*34+34] > maxValue:
                maxValue = output[i+j*34+34]
                index = i
            #print(i+j*34+31)
        print(index, maxValue)
        label = label + PLATE_CHARS_LETTER[index]
    return label

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 32, 3)
        self.conv4 = nn.Conv2d(32, 10, 1)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(28 * 7 * 10, 512)
        self.fc2 = nn.Linear(512, 238)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.leaky_relu(self.conv2(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.leaky_relu(self.conv3(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.leaky_relu(self.conv4(x))
        x = F.dropout(x)
        x = x.view(-1, 28 * 7 * 10)
        x = F.leaky_relu(self.fc1(x))
        x = F.dropout(x)
        x = self.fc2(x)
        x = x.view(-1, 7, 34)
        x = F.softmax(x, dim=2)
        x = x.view(-1, 238)
        return x





def main():
    pics = os.listdir(PICS_PATH)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = Net().to(device)
    model.load_state_dict(torch.load("car_plate.pt"))
    model.eval()

    right_count = 0
    for i in range(len(pics)):
        label = pics[i][0:7]
        img = cv.imread(PICS_PATH + "/" + label + ".jpg")
        r, g, b = cv.split(img)
        numpy_array = np.array([r, g, b])

        img_tensor = torch.from_numpy(numpy_array)
        img_tensor = img_tensor.float()
        img_tensor = img_tensor.reshape([1, 3, 70, 238])
        img_tensor = img_tensor.to(device)
        output = model(img_tensor).cpu()
        output_label = parseOutput(output.detach().numpy())
        if output_label == label:
            right_count += 1
        print("label is " + label + " ,network predict is " + output_label)
    print(len(pics), right_count)
    print("correct rate is " + str(right_count / len(pics)))


if __name__ == '__main__':
    main()