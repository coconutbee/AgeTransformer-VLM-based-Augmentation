import sys
import argparse 
import logging  
import os  
import cv2 
import torch 

from MiVOLO.mivolo.data.data_reader import InputType, get_all_files 
from MiVOLO.mivolo.predictor import Predictor
from timm.utils import setup_default_logging  
from enum import Enum


_logger = logging.getLogger("inference")

def get_parser():  
    parser = argparse.ArgumentParser(description="PyTorch MiVOLO Inference")
    parser.add_argument("--input", type=str, default=None, required=True, help="image file or folder with images")
    parser.add_argument("--detector-weights", type=str, default=None, required=True, help="Detector weights (YOLOv8).") 
    parser.add_argument("--checkpoint", default="", type=str, required=True, help="path to mivolo checkpoint")
    parser.add_argument(
        "--with-persons", action="store_true", default=False, help="If set model will run with persons, if available"  
    )
    parser.add_argument(
        "--disable-faces", action="store_true", default=False, help="If set model will use only persons if available"  
    )
    parser.add_argument("--draw", action="store_true", default=False, help="If set, resulted images will be drawn") 
    parser.add_argument("--device", default="cuda", type=str, help="Device (accelerator) to use.")  
    return parser

IMAGES_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')  

class InputType(Enum):
    Image = 0
    ImageList = 1

def get_input_type(input_path: str) -> InputType:
    if os.path.isdir(input_path):
        print(f"Input is a folder: {input_path}, processing images inside the folder.")
        return InputType.Image
    elif all(os.path.isfile(p) for p in input_path.split()):
        print(f"Input contains multiple images: {input_path}")
        return InputType.ImageList
    elif os.path.isfile(input_path):
        if input_path.endswith(IMAGES_EXT):
            return InputType.Image
        else:
            raise ValueError(f"Unknown or unsupported file format for {input_path}. Supported image formats: {IMAGES_EXT}")
    else:
        raise ValueError(f"Unknown input {input_path}")

def get_image_files(input_path: str) -> list:
    if os.path.isdir(input_path):
        image_files = []
        for root, _, files in os.walk(input_path):
            for filename in files:
                if filename.lower().endswith(IMAGES_EXT):
                    image_files.append(os.path.join(root, filename))
        return image_files
    elif all(os.path.isfile(p) for p in input_path.split()):
        return input_path.split()
    else:
        raise ValueError(f"Unknown or unsupported input path: {input_path}")



    
def main(): 
    parser = get_parser()  
    setup_default_logging()  
    args = parser.parse_args()  

    if torch.cuda.is_available(): 
        torch.backends.cuda.matmul.allow_tf32 = True  
        torch.backends.cudnn.benchmark = True  

    predictor = Predictor(args, verbose=True)  
    input_type = get_input_type(args.input)  

    ages_genders = [] 

    if input_type == InputType.Image: 
        image_files = get_all_files(args.input) if os.path.isdir(args.input) else [args.input] 

        for img_p in image_files:  
            img = cv2.imread(img_p) 
            detected_objects, out_im = predictor.recognize(img)  
            for age, gender in zip(detected_objects.ages, detected_objects.genders):
                if age is not None:
                    ages_genders.append((os.path.abspath(img_p), age, gender))

            if args.draw:  
                bname = os.path.splitext(os.path.basename(img_p))[0]  
                filename = f"out_{bname}.jpg"  
                cv2.imwrite(filename, out_im)  
                _logger.info(f"Saved result to {filename}") 


    return ages_genders

if __name__ == "__main__": 
    results = main() 
    print(results)
