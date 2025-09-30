import sys
import argparse  # 引入argparse模組，用於解析命令行參數
import logging  # 引入logging模組，用於記錄日誌信息
import os  # 引入os模組，用於操作系統相關功能（如創建文件夾）
import cv2  # 引入OpenCV模組，用於圖像和視頻處理
import torch  # 引入PyTorch模組，用於深度學習模型的開發和運行
mivolo_directory = '/media/avlab/disk1/Jim/MiVOLO'
if mivolo_directory not in sys.path:
    sys.path.insert(0, mivolo_directory)
from mivolo.data.data_reader import InputType, get_all_files  # 從mivolo.data.data_reader中引入InputType和get_all_files
from mivolo.predictor import Predictor  # 引入Predictor類，用於進行推理
from timm.utils import setup_default_logging  # 引入setup_default_logging，用於設置默認的日誌配置
from enum import Enum


_logger = logging.getLogger("inference")  # 創建一個名為"inference"的日誌記錄器

def get_parser():  # 定義get_parser函數，用於解析命令行參數
    parser = argparse.ArgumentParser(description="PyTorch MiVOLO Inference")  # 創建ArgumentParser對象
    parser.add_argument("--input", type=str, default=None, required=True, help="image file or folder with images")  # 添加"--input"參數，用於指定輸入文件或文件夾
    parser.add_argument("--detector-weights", type=str, default=None, required=True, help="Detector weights (YOLOv8).")  # 添加"--detector-weights"參數，用於指定檢測器權重文件
    parser.add_argument("--checkpoint", default="", type=str, required=True, help="path to mivolo checkpoint")  # 添加"--checkpoint"參數，用於指定mivolo模型的檢查點
    parser.add_argument(
        "--with-persons", action="store_true", default=False, help="If set model will run with persons, if available"  # 添加"--with-persons"參數，是否包括人員的檢測
    )
    parser.add_argument(
        "--disable-faces", action="store_true", default=False, help="If set model will use only persons if available"  # 添加"--disable-faces"參數，是否禁用面部檢測
    )
    parser.add_argument("--draw", action="store_true", default=False, help="If set, resulted images will be drawn")  # 添加"--draw"參數，是否繪製結果圖像
    parser.add_argument("--device", default="cuda", type=str, help="Device (accelerator) to use.")  # 添加"--device"參數，用於指定運行設備
    return parser  # 返回解析器

IMAGES_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')  # 根據需要添加擴展名

class InputType(Enum):
    Image = 0
    ImageList = 1  # 針對多個圖片的情況

def get_input_type(input_path: str) -> InputType:
    # 如果是資料夾
    if os.path.isdir(input_path):
        print(f"Input is a folder: {input_path}, processing images inside the folder.")
        return InputType.Image
    # 如果是多個圖片文件
    elif all(os.path.isfile(p) for p in input_path.split()):
        print(f"Input contains multiple images: {input_path}")
        return InputType.ImageList
    # 如果是單個圖片文件
    elif os.path.isfile(input_path):
        if input_path.endswith(IMAGES_EXT):
            return InputType.Image
        else:
            raise ValueError(f"Unknown or unsupported file format for {input_path}. Supported image formats: {IMAGES_EXT}")
    else:
        raise ValueError(f"Unknown input {input_path}")

# def get_image_files(input_path: str) -> list:
#     # 如果是資料夾，獲取該資料夾內的所有圖片文件
#     if os.path.isdir(input_path):
#         image_files = []
#         for filename in os.listdir(input_path):
#             if filename.endswith(IMAGES_EXT):
#                 image_files.append(os.path.join(input_path, filename))
#         return image_files
#     # 如果是多個圖片文件
#     elif all(os.path.isfile(p) for p in input_path.split()):
#         return input_path.split()  # 返回多個文件路徑
#     else:
#         raise ValueError(f"Unknown or unsupported input path: {input_path}")
def get_image_files(input_path: str) -> list:
    # 如果是資料夾，遞迴抓取所有子資料夾內的圖片
    if os.path.isdir(input_path):
        image_files = []
        for root, _, files in os.walk(input_path):  # 遞迴掃描
            for filename in files:
                if filename.lower().endswith(IMAGES_EXT):
                    image_files.append(os.path.join(root, filename))
        return image_files
    # 如果是多個圖片文件（用空格分開）
    elif all(os.path.isfile(p) for p in input_path.split()):
        return input_path.split()
    else:
        raise ValueError(f"Unknown or unsupported input path: {input_path}")



    
def main():  # 定義main函數
    parser = get_parser()  # 獲取命令行參數解析器
    setup_default_logging()  # 設置默認日誌
    args = parser.parse_args()  # 解析命令行參數

    if torch.cuda.is_available():  # 如果CUDA可用
        torch.backends.cuda.matmul.allow_tf32 = True  # 允許使用TF32加速矩陣運算
        torch.backends.cudnn.benchmark = True  # 啟用CUDNN的benchmark模式

    predictor = Predictor(args, verbose=True)  # 創建Predictor對象
    input_type = get_input_type(args.input)  # 獲取輸入數據的類型

    ages_genders = []  # 創建一個列表來存儲年齡信息

    if input_type == InputType.Image:  # 如果輸入為圖像
        image_files = get_all_files(args.input) if os.path.isdir(args.input) else [args.input]  # 獲取所有圖像文件

        for img_p in image_files:  # 遍歷每個圖像文件
            img = cv2.imread(img_p)  # 讀取圖像
            detected_objects, out_im = predictor.recognize(img)  # 使用預測器識別圖像中的物體
            for age, gender in zip(detected_objects.ages, detected_objects.genders):
                if age is not None:
                    ages_genders.append((os.path.abspath(img_p), age, gender))
                    # ages_genders.append((img_p.split("/")[-1], age, gender))  # 將圖像文件名和年齡添加到列表中

            if args.draw:  # 如果設置了"--draw"參數
                bname = os.path.splitext(os.path.basename(img_p))[0]  # 獲取圖像文件的基本名稱
                filename = f"out_{bname}.jpg"  # 設置輸出文件名
                cv2.imwrite(filename, out_im)  # 將結果保存到輸出文件
                _logger.info(f"Saved result to {filename}")  # 記錄保存結果的日誌

    # 返回年齡預測結果列表
    return ages_genders

if __name__ == "__main__":  # 如果當前文件是主程序
    results = main()  # 調用main函數並獲取結果
    # 主程式可以在此處統一處理結果（例如寫入文件等）
    print(results)  # 打印預測結果
