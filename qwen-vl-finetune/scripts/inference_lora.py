#!/usr/bin/env python3
# LoRA Model Inference Script
# Load base model + trained LoRA adapters for inference

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
import argparse
import os

def load_lora_model(base_model_path, lora_adapter_path):
    """Load base model and apply LoRA adapters"""
    print(f"Loading base model from: {base_model_path}")
    
    # Load base model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print(f"Loading LoRA adapters from: {lora_adapter_path}")
    # Apply LoRA adapters
    model = PeftModel.from_pretrained(model, lora_adapter_path)
    
    # Load processor
    processor = AutoProcessor.from_pretrained(base_model_path)
    
    return model, processor

def inference(model, processor, image_path, prompt, max_new_tokens=2048):
    """Run inference on a single image"""
    from PIL import Image
    
    # Load and process image
    image = Image.open(image_path)
    
    # Prepare conversation
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    # Apply chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # Process inputs
    inputs = processor(
        text=[text], 
        images=[image], 
        padding=True, 
        return_tensors="pt"
    )
    
    # Move to device
    inputs = inputs.to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.7,
            top_p=0.9
        )
    
    # Decode response
    response = processor.batch_decode(
        outputs[:, inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True
    )[0]
    
    return response

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True, help="Path to base model")
    parser.add_argument("--lora_adapter", required=False, help="Path to LoRA adapter (optional)")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--prompt", default="Describe this image in detail.", help="Input prompt")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum tokens to generate")
    
    args = parser.parse_args()
    
    # Decide whether to load LoRA adapters
    if args.lora_adapter and os.path.exists(os.path.join(args.lora_adapter, "adapter_config.json")):
        model, processor = load_lora_model(args.base_model, args.lora_adapter)
    else:
        print("No valid LoRA adapter provided. Loading base model only.")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(args.base_model)
    
    # Run inference
    response = inference(model, processor, args.image, args.prompt, args.max_new_tokens)
    
    print(f"Input: {args.prompt}")
    print(f"Response: {response}")

if __name__ == "__main__":
    main() 