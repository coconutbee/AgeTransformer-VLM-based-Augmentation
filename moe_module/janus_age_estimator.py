import sys
sys.path.append('/media/avlab/disk1/Janus')
import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from PIL import Image
import os
import argparse
from tqdm import tqdm

# Resize image to the target size
def resize_image(image_path, target_size):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
    return image

# Get image paths from a folder
def get_image_paths(folder_path):
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths

# The function to run the model and get age prediction directly
def predict_age(model_path, image_paths, prompt="Please provide a numerical age. Only answer with a number. If unable to answer, say unknown.", resize=128):
    # Load the model
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    predictions = []

    # Process each image
    for image_path in tqdm(image_paths, desc="Processing images", unit="image"):
        resized_image = resize_image(image_path, resize)
        conversation = [
            {"role": "<|User|>", "content": f"<image_placeholder>\n{prompt}", "images": [image_path]},
            {"role": "<|Assistant|>", "content": ""},
        ]
        
        pil_images = [resized_image]
        prepare_inputs = vl_chat_processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(vl_gpt.device)
        
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
        outputs = vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
        )
        
        # Decode and extract the answer
        answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        
        # Store the prediction result
        predictions.append({
            "image": os.path.basename(image_path),
            "answer": answer
        })

    return predictions

# Example usage
def main():
    parser = argparse.ArgumentParser(description="Age Prediction with Janus AI Model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model.")
    parser.add_argument("--image_folder", type=str, required=True, help="Folder containing images for prediction.")
    parser.add_argument("--resize", type=int, default=256, help="Resize dimension for images.")
    parser.add_argument("--prompt", type=str, default="Please provide a numerical age. Only answer with a number. If unable to answer, say unknown.", help="Prompt for the model.")
    
    # Parse the arguments
    args = parser.parse_args()

    # Get image paths from the provided folder
    image_paths = get_image_paths(args.image_folder)
    
    if not image_paths:
        print(f"No images found in the folder: {args.image_folder}")
        return

    # Call the model to get predictions
    predictions = predict_age(args.model_path, image_paths, prompt=args.prompt, resize=args.resize)
    
    # Print predictions for each image
    for prediction in predictions:
        print(f"Image: {prediction['image']}, Predicted Age: {prediction['answer']}")

if __name__ == "__main__":
    main()
