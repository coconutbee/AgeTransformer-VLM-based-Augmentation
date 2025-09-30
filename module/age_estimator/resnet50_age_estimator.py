
import os
import argparse

from module.age_estimator.resnet50_ft_dims_2048_new import resnet50_ft
import torch
import cv2
import numpy as np
import csv
from PIL import Image
from torch import nn

LAMBDA_1 = 0.2
LAMBDA_2 = 0.05
START_AGE = 0
END_AGE = 100

def predict_resnet50(model, image):

    model.eval()
    with torch.no_grad():
        #image = image.astype(np.float32)
      
        image = np.transpose(image, (2,0,1))
        img = torch.from_numpy(image).cuda()
        img = img.type('torch.FloatTensor').cuda()
       
        output = model(img[None])
        m = nn.Softmax(dim=1)
        
        output = m(output)
        
        a = torch.arange(START_AGE, END_AGE + 1, dtype=torch.float32).cuda()
        mean = (output * a).sum(1, keepdim=True).cpu().data.numpy()
        pred = np.around(mean)[0][0]
    return pred

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-pm', '--pred_model', type=str, default='/media/avlab/2TB_new/Age_estimator/result/New_resnet50/age_estimator.pth')
    parser.add_argument('-path', '--pred_path', type=str, default='/media/avlab/2TB_new/Janus/face')
    return parser.parse_args()

def main():
    args = get_args()

    if args.pred_path and args.pred_model:
        print(f"Loading model from: {args.pred_model}")
        model = resnet50_ft()
        model.load_state_dict(torch.load(args.pred_model, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
        model.eval()
        model.cuda()

        all_predictions = {}

        if not os.path.exists(args.pred_path):
            print(f"Error: Path '{args.pred_path}' does not exist.")
            return {}

        # 儲存 `資料夾` 內直接放的圖片預測結果
        folder_predictions = {}

        for filename in os.listdir(args.pred_path):
            img_path = os.path.join(args.pred_path, filename)

            # 檢查是否為影像檔案
            if os.path.isfile(img_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = cv2.imread(img_path)

                if img is None:
                    print(f"Warning: Unable to read image {img_path}")
                    continue

                resized_img = cv2.resize(img, (224, 224))
                pred = predict_resnet50(model, resized_img)
                folder_predictions[filename] = pred

        # 如果有預測結果，加入 `all_predictions`
        if folder_predictions:
            all_predictions["face"] = folder_predictions

        # 處理 `資料夾` 內的子資料夾
        for foldername in os.listdir(args.pred_path):
            folder_path = os.path.join(args.pred_path, foldername)

            if os.path.isdir(folder_path):
                subfolder_predictions = {}

                for filename in os.listdir(folder_path):
                    img_path = os.path.join(folder_path, filename)

                    if os.path.isfile(img_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img = cv2.imread(img_path)

                        if img is None:
                            print(f"Warning: Unable to read image {img_path}")
                            continue

                        resized_img = cv2.resize(img, (224, 224))
                        pred = predict_resnet50(model, resized_img)
                        subfolder_predictions[filename] = pred

                if subfolder_predictions:
                    all_predictions[foldername] = subfolder_predictions

        return all_predictions

if __name__ == "__main__":
    predictions = main()
    if predictions:
        print(predictions)
    else:
        print("No predictions were made.")