import sys
import os
import re
import subprocess
import time
import ast
import requests
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import requests
import cv2
import torch
import json
import csv
from module.age_estimator import janus_age_estimator
from module.age_estimator import resnet50_ft_dims_2048_new, vgg16_age_estimator, resnet50_age_estimator
from module.age_estimator.resnet50_age_estimator import resnet50_ft
from module.age_estimator.resnet50_age_estimator import predict_resnet50
from module.age_estimator.vgg16_age_estimator import VGG16_AGE, predict_vgg16, write_batch_to_csv
from PIL import Image
from torch import nn
import logging
import warnings

# Setup logging
logging.basicConfig(filename='warnings.log', level=logging.WARNING)
warnings.simplefilter("always")
logging.captureWarnings(True)

CONFIG = {
    'image_folder': "./data/test_images",  
    'janus_model_path': "deepseek-ai/Janus-Pro-7B",  
    'mivolo_checkpoint': "./models/model_imdb_cross_person_4.22_99.46.pth.tar",  
    'vgg16_model_path': "./models/model_best_loss",  
    'resnet_model_path': "./models/age_estimator.pth",  
    'moe_model_path': "./models/best_dynamic_moe_model.pth",
    'start_age': 0,  
    'end_age': 120  
}

# Device check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model columns
model_columns = ["Janus-7B-Pro", "Vgg16", "MiVOLO", "Resnet50"]

# Run Janus in conda env
def run_janus_model(image_path):
    result = subprocess.run(['conda', 'run', '-n', 'Janus', 'python', 'janus_model.py', '--input', image_path], capture_output=True, text=True)
    return result.stdout

# Run MiVOLO in conda env
def run_mivolo_model(image_path):
    result = subprocess.run(['conda', 'run', '-n', 'mivolo', 'python', 'mivolo_model.py', '--input', image_path], capture_output=True, text=True)
    return result.stdout

# Run VGG16 in conda env
def run_vgg16_model(image_path):
    result = subprocess.run(['conda', 'run', '-n', 'IPCGAN', 'python', 'vgg16_model.py', '--input', image_path], capture_output=True, text=True)
    return result.stdout

# Run ResNet50 in conda env
def run_Resnet50_model(image_path):
    result = subprocess.run(['conda', 'run', '-n', 'IPCGAN', 'python', 'Resnet50_model.py', '--input', image_path], capture_output=True, text=True)
    return result.stdout


# Expert sub-network
class ExpertMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.mlp(x)


# Mixture of Experts (MoE) Network
class MixtureOfExpertsNet(nn.Module):
    def __init__(self, num_experts=4):
        super(MixtureOfExpertsNet, self).__init__()
        self.num_experts = num_experts
        self.gating = nn.Linear(num_experts, num_experts)
        self.expert_mlps = nn.ModuleList([ExpertMLP() for _ in range(num_experts)])

    def forward(self, x):
        mask = ~torch.isnan(x)
        x_filled = torch.where(mask, x, torch.zeros_like(x))

        adjusted = []
        for i in range(self.num_experts):
            expert_input = x_filled[:, i].view(-1, 1)
            adjusted_output = self.expert_mlps[i](expert_input).squeeze(1)
            adjusted_output = torch.clamp(adjusted_output, min=0)
            adjusted.append(adjusted_output)
        adjusted_experts = torch.stack(adjusted, dim=1)

        gating_logits = self.gating(x_filled)
        gating_weights = torch.softmax(gating_logits, dim=1)
        gating_weights = gating_weights * mask.float()
        weight_sum = gating_weights.sum(dim=1, keepdim=True)
        gating_weights = torch.where(weight_sum > 0, gating_weights / weight_sum, gating_weights)

        prediction = torch.sum(gating_weights * adjusted_experts, dim=1)
        prediction = torch.where(weight_sum.squeeze() > 0, prediction, torch.full_like(prediction, float('nan')))
        return prediction


# Collect image paths
def get_image_paths(image_folder):
    image_paths = []
    for root, _, files in os.walk(image_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    return image_paths


# Janus age estimator
def run_janus_age_estimator(image_paths, model_path=CONFIG['janus_model_path'], prompt="Please provide a numerical age. Only answer with a number. If unable to answer, say unknown.", resize=256):
    janus_script_path = "./module/age_estimator/janus_age_estimator.py"
    command = [
        "python", janus_script_path, 
        "--model_path", model_path, 
        "--image_folder", CONFIG['image_folder'], 
        "--prompt", prompt, 
        "--resize", str(resize)
    ]
    predictions = {}

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output = result.stdout.strip().split('\n')

        if len(output) != len(image_paths):
            print(f"Warning: Output lines ({len(output)}) do not match image count ({len(image_paths)})")

        for line, img_path in tqdm(zip(output, image_paths), total=len(image_paths), desc="Processing Janus", unit="image"):
            match = re.search(r"Predicted Age:\s*(\d+|unknown)", line)
            if match:
                age = match.group(1)
                predictions[img_path] = float(age) if age.isdigit() else "unknown"
            else:
                predictions[img_path] = "unknown"

    except subprocess.TimeoutExpired:
        for img_path in image_paths:
            predictions[img_path] = "timeout"

    except subprocess.CalledProcessError as e:
        print(f"Error during Janus age prediction: {e}")
        print(f"Error output: {e.stderr}")
        for img_path in image_paths:
            predictions[img_path] = "unknown"

    except Exception as e:
        print(f"Unexpected error: {e}")
        for img_path in image_paths:
            predictions[img_path] = "unknown"

    return predictions


# MiVOLO age estimator
def run_mivolo_age_estimator(image_folder=CONFIG['image_folder'], checkpoint=CONFIG['mivolo_checkpoint']):
    mivolo_script_path = "./module/age_estimator/mivolo_age_estimator.py"
    command = [
        "conda", "run", "-n", "mivolo", "python", mivolo_script_path,
        "--input", image_folder, "--checkpoint", checkpoint,
        "--detector-weights", "./checkpoint/yolov8x_person_face.pt"
    ]
    predictions = {}

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output_lines = result.stdout.strip().split("\n")
        try:
            predictions_list = ast.literal_eval(output_lines[-1])
        except (SyntaxError, ValueError):
            print("Error: Unable to parse MiVOLO output.")
            return predictions  

        if not isinstance(predictions_list, list):
            print("Error: Parsed data is not a list.")
            return predictions  

        for item in tqdm(predictions_list, desc="Processing Images", unit="image"):
            if isinstance(item, tuple) and len(item) == 3:
                img_path, age, gender = item
                if os.path.isabs(img_path):
                    full_path = os.path.abspath(img_path)
                else:
                    full_path = os.path.abspath(os.path.join(image_folder, img_path))
                predictions[full_path] = float(age)
        return predictions

    except subprocess.CalledProcessError as e:
        print(f"Error executing script: {e.stderr}")
    except subprocess.TimeoutExpired:
        print("Execution timed out")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
    
    return predictions  


# VGG16 age estimator
def run_vgg16_age_estimator(image_paths, model):
    predictions = {}
    for image_path in tqdm(image_paths, desc="VGG16"):
        image = cv2.imread(image_path)
        if image is None:
            predictions[image_path] = np.nan
            continue
        resized_image = cv2.resize(image, (224, 224))
        pred = predict_vgg16(model, resized_image)
        predictions[image_path] = pred if pred is not None else np.nan
    return predictions


# ResNet50 age estimator
def run_resnet_age_estimator(image_paths, model):
    predictions = {}
    for image_path in tqdm(image_paths, desc="Resnet50"):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error loading image: {image_path}")
            predictions[image_path] = np.nan
            continue
        resized_img = cv2.resize(img, (224, 224))
        pred = predict_resnet50(model, resized_img)
        predictions[image_path] = pred if pred is not None else np.nan
    return predictions


# Save results to CSV
def write_predictions_to_csv(image_paths, predictions_dicts, output_filename):
    all_models = list(predictions_dicts.keys())
    columns = ['Image'] + all_models

    rows = []
    for img_path in image_paths:
        row = {'Image': img_path}
        for model_name in all_models:
            model_preds = predictions_dicts[model_name]
            row[model_name] = model_preds.get(img_path, np.nan)
        rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_filename, index=False)
    print(f"Results saved to {output_filename}")


# Main
def main():
    start_time = time.time()
    image_paths = get_image_paths(CONFIG['image_folder'])
    if not image_paths:
        print(f"No images found in the folder: {CONFIG['image_folder']}")
        return
    
    print("Running Janus prediction...")
    predictions_from_janus = run_janus_age_estimator(image_paths)
    output_filename = f'predictions_Janus.csv'

    with open(output_filename, 'w', newline='') as csvfile:
        fieldnames = ['Image', 'Predicted_Age']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for img_path, age in predictions_from_janus.items():
            writer.writerow({'Image': img_path, 'Predicted_Age': age})

    print(f"Results saved to {output_filename}")

    print("Running MiVOLO prediction...")
    predictions_from_mivolo = run_mivolo_age_estimator(image_folder=CONFIG['image_folder'])

    vgg16_model = VGG16_AGE()
    vgg16_model.load_state_dict(torch.load(CONFIG['vgg16_model_path'], weights_only=True))
    vgg16_model.eval()  
    vgg16_model.to(device)
    predictions_from_vgg16 = run_vgg16_age_estimator(image_paths, vgg16_model)
    write_batch_to_csv(predictions_from_vgg16.items(), "VGG16_AGE")

    print("Running ResNet50 prediction...")
    resnet_model = resnet50_ft()
    resnet_model.load_state_dict(torch.load(CONFIG['resnet_model_path'], weights_only=True))
    resnet_model.eval()
    resnet_model.to(device)
    predictions_from_resnet50 = run_resnet_age_estimator(image_paths, resnet_model)

    df = pd.DataFrame({
        'Image': image_paths,
        'Janus-7B-Pro': [predictions_from_janus.get(img, np.nan) for img in image_paths],
        'MiVOLO': [predictions_from_mivolo.get(img, np.nan) for img in image_paths],
        'Vgg16': [predictions_from_vgg16.get(img, np.nan) for img in image_paths],
        'Resnet50': [predictions_from_resnet50.get(img, np.nan) for img in image_paths]
    })

    df[model_columns] = df[model_columns].apply(pd.to_numeric, errors="coerce")
    df[model_columns] = df[model_columns].where(df[model_columns] <= 150, "error")
    df.replace(["error", "Unknown"], np.nan, inplace=True)
    df[model_columns] = df[model_columns].apply(pd.to_numeric, errors='coerce')

    print("Running Mixture of Experts prediction...")
    moe_input_tensor = torch.tensor(df[model_columns].values.astype(np.float32)).to(device)

    moe_model = MixtureOfExpertsNet(num_experts=len(model_columns)).to(device)
    moe_model.load_state_dict(torch.load(CONFIG['moe_model_path'], map_location=device))
    moe_model.eval()

    with torch.no_grad():
        moe_preds = moe_model(moe_input_tensor).detach().cpu().numpy()

    df["MoE_Predicted_Age"] = moe_preds
    df[["Image"] + model_columns + ["MoE_Predicted_Age"]].to_csv("predictions.csv", index=False)
    print("Inference completed, results saved to predictions.csv")

    total_time = time.time() - start_time
    print(f"Average processing time per image: {total_time / len(image_paths):.2f} seconds")

if __name__ == "__main__":
    main()
