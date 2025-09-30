import os 
import time 
import json 
import argparse
import torch 
import torchvision
import random
import numpy as np 
from tqdm import tqdm 
from torch import nn
from collections import OrderedDict
from torchvision.models.resnet import resnet18
import sys
sys.path.append('/media/avlab/disk1/Jim/age_estimator')
from module.age_estimator.resnet50_ft_dims_2048_new import resnet50_ft
from module.age_estimator.mean_variance_loss import MeanVarianceLoss
import cv2
import torch.nn.functional as F
import csv
from PIL import Image
LAMBDA_1 = 0.2
LAMBDA_2 = 0.05
START_AGE = 0
END_AGE = 100
VALIDATION_RATE= 0.1

# 批量寫入CSV函數
def write_batch_to_csv(predictions, model_name):
    with open('predictions.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        for filename, pred in predictions:
            writer.writerow([filename, model_name, pred])  # 寫入每一個預測結果

def vgg_block(in_channels, out_channels, more=False):
    blocklist = [
        ('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)),
        ('relu1', nn.ReLU(inplace=True)),
        ('conv2', nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)),
        ('relu2', nn.ReLU(inplace=True)),
    ]
    if more:
        blocklist.extend([
            ('conv3', nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)),
            ('relu3', nn.ReLU(inplace=True)),
        ])
    blocklist.append(('maxpool', nn.MaxPool2d(kernel_size=2, stride=2)))
    block = nn.Sequential(OrderedDict(blocklist))
    return block

class VGG(nn.Module):
    def __init__(self, classes=1000, channels=3):
        super().__init__()
        self.conv = nn.Sequential(
            vgg_block(channels, 64),
            vgg_block(64, 128),
            vgg_block(128, 256, True),
            vgg_block(256, 512, True),
            vgg_block(512, 512, True),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5, inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5, inplace=True),
        )
        self.cls = nn.Linear(4096, 101)

    def forward(self, x):
        in_size = x.shape[0]
        x = self.conv(x)
        x = x.view(in_size, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.cls(x)
        # x = F.softmax(x, dim=1)
        return x
class VGG16_AGE(VGG):
    def __init__(self, classes=101, channels=3):
        super().__init__()
        self.cls = nn.Linear(4096, 101)

def ResNet18(num_classes):

    model = resnet18(pretrained=True)
    model.fc = nn.Sequential(
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes),
    )
    return model

def predict_vgg16(model, image):
    model.eval()
    with torch.no_grad():
        # 轉換影像通道順序
        image = np.transpose(image, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        img = torch.from_numpy(image).float().cuda()
        
        # 標準化 (這行可能需要視訓練時是否使用)
        mean = torch.tensor([0.485, 0.456, 0.406]).cuda()
        std = torch.tensor([0.229, 0.224, 0.225]).cuda()
        img = (img - mean[:, None, None]) / std[:, None, None]

        img = img[None]  # 增加 batch 維度
        
        output = model(img)
        output = F.softmax(output, dim=1)  # 確保 softmax 正確應用
        
        a = torch.arange(START_AGE, END_AGE + 1, dtype=torch.float32).cuda()
        mean = (output * a).sum(1, keepdim=True).cpu().data.numpy()
        pred = np.around(mean)[0][0]
    return pred

def get_image_list(image_directory):
    image_paths = []
    for filename in os.listdir(image_directory):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # Add more formats if needed
            image_paths.append(os.path.join(image_directory, filename))
    return image_paths


def process_images(image_directory, model_path, model_name):
    model = VGG16_AGE()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.cuda()

    predictions = []  # 用來儲存所有預測結果

    # 遍歷資料夾中的每張圖片並進行預測
    for filename in os.listdir(image_directory):
        img = cv2.imread(os.path.join(image_directory, filename))
        
        if img is None:
            print(f"Error loading image {filename}")
            continue  # Skip the image if it's not loaded correctly
        
        resized_img = cv2.resize(img, (224, 224))  # Resize to 224x224 for VGG16

        pred = predict_vgg16(model, resized_img)
        
        # 將圖片的檔名與預測年齡存入結果列表
        predictions.append((filename, pred))

        # 每處理一批圖片後就寫入一次CSV
        if len(predictions) >= 100:  # 假設每次處理100張圖片後寫入一次
            write_batch_to_csv(predictions, model_name)
            predictions = []  # 清空 predictions，為下一批預測準備

    # 如果還有剩餘的預測結果，寫入CSV
    if predictions:
        write_batch_to_csv(predictions, model_name)

    return predictions

def main():
    image_directory = "/media/avlab/2TB_new/Janus/ffhq_add_yuri"  # 替換為您的圖片資料夾路徑
    model_path = "/media/avlab/2TB_new/Age_estimator/result/mean_variance_ffhq/model_best_loss"  # 替換為您的模型路徑

    # 處理圖片並獲取預測結果
    model_name = "VGG16_AGE"
    predictions = process_images(image_directory, model_path, model_name)

    # 顯示預測結果
    # for filename, pred in predictions:
    #     print(f"Image: {filename}, Predicted Age: {pred}")

if __name__ == "__main__":
    main()
